# Feature extraction and Data Analysis

import os
import cv2
import numpy as np

def extract_features(data_dir):
    classes = ["RealArt", "AIArtData"]
    X_features = [] 
    y_labels = [] # 0-RealArt, 1-AIArtData
    
    # extract features for each class
    for label, class_label in enumerate(classes): # gives index and class label
        class_path = os.path.join(data_dir, class_label)

        if not os.path.exists(class_path):
            print(f"Error: {class_path} does not exist")
            continue

        # loop through all images in the class folder
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(class_path, img_file) # create full path to image
            img = cv2.imread(img_path)

            if img is None: # if image is corrupted skip it
                continue

            img = cv2.resize(img, (256, 256)) # standardize image size for HOG

            # Color Histogram
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist_features = []
            for i in range(3):
                if i == 0: # hue channel
                    ranges = [0, 180]
                else: # saturation and value channels
                    ranges = [0, 256]
                hist = cv2.calcHist([img_hsv], [i], None, [32], ranges)
                hist = cv2.normalize(hist, hist).flatten()
                hist_features.extend(hist)

            # Combine features
            X_features.append(hist_features)
            y_labels.append(label) # 0-RealArt, 1-AIArtData

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)

    np.save("extracted_features/X_features.npy", X_features)
    np.save("extracted_features/y_labels.npy", y_labels)
    print(f"Features saved to extracted_features/X_features.npy and extracted_features/y_labels.npy")

if __name__ == "__main__":
    DATA_DIR = "./Art"
    extract_features(DATA_DIR)