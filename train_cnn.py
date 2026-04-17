import tensorflow as tf
import keras
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
import os

# load data
data_dir = "./Art"
batch_size = 32
img_size = (224, 224)

# Create the training dataset with random flips and rotations to prevent overfitting
# Update your training and validation loaders to look like this:
train_dataset = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    crop_to_aspect_ratio=True  # <--- ADD THIS LINE
)
# Augmentation
augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomBrightness(0.1),
])

# Force Keras to generate augmented images on the fly during training, instead of storing them all in memory
train_dataset = train_dataset.map(lambda x, y: (augmentation(x, training=True), y))

# Create the validation (testing) dataset (No augmentation here)
val_dataset = keras.utils.image_dataset_from_directory(data_dir,validation_split=0.2,subset="validation",seed=42,image_size=img_size,batch_size=batch_size,label_mode='binary')

# Optimize data loading speed
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Build the architecture
# Load ResNet50, but chop off the head
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Phase 1: Freeze the base model completely
base_model.trainable = False

# Attach our custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x) # Flatten the 2D maps
x = Dense(256, activation='relu')(x) # A smart processing layer
x = Dropout(0.5)(x) # Drop 50% of connections randomly to prevent overfitting
predictions = Dense(1, activation='sigmoid')(x) # Output: 0 (Real) or 1 (AI)

# Stitch it all together into one massive model
model = Model(inputs=base_model.input, outputs=predictions)

# Phase 1: warm up the head
print("\nWarming up the custom head...")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])

# Train just the head for a few epochs
history_phase1 = model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# Phase 2: fine-tune the CNN
print("\nUnfreezing the CNN for Fine-Tuning...")
# Unfreeze the base model
base_model.trainable = True

# Freeze the bottom 100 layers (the ones that find basic lines and edges)
# We only want to retrain the top layers (the ones that find complex textures)
for layer in base_model.layers[:100]:
    layer.trainable = False

# recompile with a microscopic learning rate so we don't destroy the brain
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),loss='binary_crossentropy',metrics=['accuracy'])

# Train the unfrozen network
history_phase2 = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Save the best model
print("\nSaving the model...")
model.save('saved_models/ai_art_classifier_resnet.keras')