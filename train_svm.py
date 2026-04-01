import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from svm import SVM

def train_svm_model():
    X=np.load("extracted_features/X_features.npy")
    y=np.load("extracted_features/y_labels.npy")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm_model = SVM(learning_rate=0.00001, lambda_param=0.6, iterations=1000)
    svm_model.fit(X_train, y_train)

    joblib.dump(svm_model, "saved_models/svm_model.joblib")
    joblib.dump(scaler, "saved_models/scaler.joblib")

    y_pred = svm_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.3f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real Art", "AI Art"]) + "\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    train_svm_model()