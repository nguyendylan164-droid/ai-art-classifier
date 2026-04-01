import os
import cv2 
import numpy as np
import keras
from keras.applications import VGG16, ResNet50, EfficientNetB0
from keras.applications.vgg16 import preprocess_input

def extract_cnn_features(data_dir):
    classes = ["RealArt", "AIArtData"]
    X_features = [] 
    y_labels = [] # 0-RealArt, 1-AIArtData

    #feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    for label, class_label in enumerate(classes):
        class_path = os.path.join(data_dir, class_label)

        if not os.path.exists(class_path):
            print(f"Error: {class_path} does not exist")
            continue

        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (224, 224))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_batch = np.expand_dims(img_rgb, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            cnn_features = feature_extractor.predict(img_preprocessed, verbose=0)

            X_features.append(cnn_features.flatten())
            y_labels.append(label)

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)

    np.save("extracted_features/X_features.npy", X_features)
    np.save("extracted_features/y_labels.npy", y_labels)
    print(f"CNN features saved to X_features.npy and y_labels.npy")

if __name__ == "__main__":
    DATA_DIR = "./Art"
    extract_cnn_features(DATA_DIR)