# Data Analysis

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

            # Feature 1: Edge Density
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
            edges = cv2.Canny(gray, 100, 200) # detect edges
            # calculate edge density by counting white pixels and dividing by total number of pixels
            edge_density = np.sum(edges>0) / (edges.shape[0] * edges.shape[1])

            # Feature 2: Color Histogram
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to OpenCV's default BGR format to RGB format
            hist_features =[]

            for i in range(3): # loop throught the 3 channels (R, G, B)
                hist = cv2.calcHist([img_rgb], [i], None, [32], [0, 256]) # count the pixels into 32 bins
                hist = cv2.normalize(hist, hist).flatten() # normalize the histogram
                hist_features.extend(hist)

            # Combine features
            combined_features = [edge_density] + hist_features
            X_features.append(combined_features)
            y_labels.append(label) # 0-RealArt, 1-AIArtData

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)

    np.save("X_features.npy", X_features)
    np.save("y_labels.npy", y_labels)
    print(f"Features saved to X_features.npy and y_labels.npy")

if __name__ == "__main__":
    DATA_DIR = "./Art"
    extract_features(DATA_DIR)