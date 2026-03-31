import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

# extract Color Histogram
# extract HOG features
# extract LBP features
# extract laplacian variance (measure of blurriness)

def extract_features(data_dir):
    classes = ["RealArt", "AIArtData"]
    X_features = []
    y_labels = [] # 0-RealArt, 1-AIArtData

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

            # standardize image size for HOG
            img = cv2.resize(img, (128, 128)) # maybe change later

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Color Histogram
            hist_features = []
            for i in range(3):
                hist = cv2.calcHist([img_rgb], [i], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hist_features.extend(hist)

            # HOG
            hog_feats = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
            # LBP
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= (lbp_hist.sum() + 1e-7)

            # Laplacian Variance
            blur_score = [cv2.Laplacian(gray, cv2.CV_64F).var()]

            combined = hist_features + hog_feats.tolist() + lbp_hist.tolist() + blur_score

            X_features.append(combined)
            y_labels.append(label)

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)
    np.save("X_features.npy", X_features)
    np.save("y_labels.npy", y_labels)

if __name__ == "__main__":
    DATA_DIR = "./Art"
    extract_features(DATA_DIR)