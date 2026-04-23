"""Train a binary RealArt vs AIArtData CNN with transfer learning.

Pipeline overview:
1) Load a reproducible train/validation split from Art/.
2) Apply augmentation and ResNet50 preprocessing.
3) Warm up the classification head with frozen backbone.
4) Unfreeze part of the backbone for low-LR fine-tuning.
5) Save checkpoints and final model under saved_models/.
"""

import os

import numpy as np
import tensorflow as tf
import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from sklearn.utils.class_weight import compute_class_weight

# load data
data_dir = "./Art"
batch_size = 32
img_size = (224, 224)
split_seed = 123
phase1_epochs = 10
phase2_epochs = 10


def training_split_dataset():
    """Create the training split once so class weights and training use same split."""
    return keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=split_seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        crop_to_aspect_ratio=True,
    )


# Balanced class weights from a separate pass (do not consume the training pipeline)
labels_for_weights = []
for _, y in training_split_dataset():
    labels_for_weights.append(y.numpy())
y_train_flat = np.concatenate(labels_for_weights).ravel().astype(int)
classes = np.unique(y_train_flat)
cw = compute_class_weight("balanced", classes=classes, y=y_train_flat)
class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
print(f"Class weights (for balanced loss): {class_weight}")

weight_table = tf.constant(
    [class_weight.get(0, 1.0), class_weight.get(1, 1.0)], dtype=tf.float32
)


def add_sample_weight(x, y):
    """Attach per-example class weights so minority class contributes more to loss."""
    y_int = tf.cast(tf.reshape(y, [-1]), tf.int32)
    sw = tf.gather(weight_table, y_int)
    return x, y, sw


train_dataset = training_split_dataset()

augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomBrightness(0.1),
    ]
)

AUTOTUNE = tf.data.AUTOTUNE
# Cache decoded images before augmentation so random augmentations differ each epoch
train_dataset = (
    train_dataset.cache()
    # Augment before preprocess_input so model sees realistic image variation.
    .map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    # Match ImageNet preprocessing expected by ResNet50 backbone.
    .map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    .map(add_sample_weight, num_parallel_calls=AUTOTUNE)
    .prefetch(buffer_size=AUTOTUNE)
)

val_dataset = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=split_seed,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary",
    crop_to_aspect_ratio=True,
)
val_dataset = (
    val_dataset.cache()
    # Validation must use the same preprocessing as training/inference.
    .map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    .prefetch(buffer_size=AUTOTUNE)
)

os.makedirs("saved_models", exist_ok=True)

# Build a transfer-learning model: ResNet50 backbone + small binary head.
base_model = ResNet50(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

callbacks_phase1 = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="saved_models/ai_art_classifier_resnet_phase1.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

callbacks_phase2 = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="saved_models/ai_art_classifier_resnet.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

print("\nWarming up the custom head...")
# Phase 1: train only the new head so it adapts before full fine-tuning.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=phase1_epochs,
    callbacks=callbacks_phase1,
)

print("\nUnfreezing the CNN for Fine-Tuning...")
base_model.trainable = True

# Keep early layers frozen (generic edge/texture features), tune deeper layers.
for layer in base_model.layers[:100]:
    layer.trainable = False

# Phase 2: fine-tune trainable backbone layers with a much smaller LR.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=phase2_epochs,
    callbacks=callbacks_phase2,
)

print("\nSaving the model (best weights already restored by EarlyStopping; checkpoint is best val)...")
model.save("saved_models/ai_art_classifier_resnet.keras")
