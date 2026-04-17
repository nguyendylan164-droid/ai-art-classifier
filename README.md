# ai-art-classifier

Classify images as real art vs. AI-generated art using a CNN, with an optional hand-crafted feature + SVM path.

## Prerequisites

- Python 3.9+ recommended
- A GPU is optional but speeds up CNN training

## 1. Clone and create a virtual environment

```bash
cd ai-art-classifier
python -m venv .venv
```

**Activate the environment**

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **macOS / Linux:** `source .venv/bin/activate`

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Prepare your dataset

Download the AI vs. Human Art dataset from [Kaggle Link](https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art). Extract the zip file and place the AiArtData and RealArt folders inside an /Art directory at the root of this project

The CNN trainer (`train_cnn.py`) and feature extraction (`feature_extraction.py`) both expect this layout.

## 4. Train the CNN

```bash
python train_cnn.py
```

This may take up to 10 minutes

This writes the trained model to `saved_models/ai_art_classifier_resnet.keras`. Training downloads ResNet50 ImageNet weights on first run.

## 5. Run the web API

After the CNN is trained and the file above exists:

```bash
python app.py
```

The Flask server starts (default `http://127.0.0.1:5000`). Send a **POST** request to `/predict` with multipart form field **`image`** (the image file).

## Optional: SVM pipeline

Hand-crafted color histogram features and a custom SVM:

1. Extract features:

   ```bash
   python feature_extraction.py
   ```

   This creates `extracted_features/X_features.npy` and `extracted_features/y_labels.npy`.

2. Train the SVM:

   ```bash
   python train_svm.py
   ```

   This saves `saved_models/svm_model.joblib` and `saved_models/scaler.joblib`.

The REST app in `app.py` uses the **CNN** model, not the SVM.

## Troubleshooting

- **`FileNotFoundError` for `saved_models/ai_art_classifier_resnet.keras`:** Run `train_cnn.py` first (step 4).
- **Empty or wrong classes:** Ensure `Art/` has exactly the two class folders with images inside them.
- **TensorFlow install issues:** See the [official TensorFlow install guide](https://www.tensorflow.org/install) for your OS and Python version. May need to create venv with Python 3.11.