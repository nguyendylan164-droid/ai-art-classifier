# Hyperparameter Tuning Summary

## svm
- Best CV F1: 0.7340
- Best params: {'model__C': 1, 'model__gamma': 'scale', 'model__kernel': 'rbf'}

## random_forest
- Best CV F1: 0.7570
- Best params: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 400}

## custom_svm
- Implementation in `svm.py` (gradient-descent SVM), hyperparameters: learning_rate=1e-05, lambda_param=0.6, iterations=1000, with `StandardScaler` on the same train split.

## cnn
- Not tuned in this script; train with `train_cnn.py` (ResNet50 + head). Evaluated here on the **same** stratified test image set as SVM / Random Forest.
