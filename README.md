# ai-art-classifier

Classify images as real art vs. AI-generated art using a CNN, with feature-based classical ML baselines and a rubric-ready analysis workflow.

## Prerequisites

- Python 3.10-3.12 recommended (TensorFlow compatibility)
- A GPU is optional but speeds up CNN training

## 1. Clone and create a virtual environment

Use Python 3.10 on Windows for best TensorFlow compatibility.

```bash
cd ai-art-classifier
py -0p
py -3.10 -m venv .env
```

**Activate the environment**

- **Windows (PowerShell):** `.env\Scripts\Activate.ps1`
- **Windows (cmd):** `.env\Scripts\activate.bat`
- **macOS / Linux:** `source .env/bin/activate`

Then confirm Python points to 3.10:

```bash
python --version
```

## 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional TensorFlow sanity check:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 3. Prepare your dataset

Download the AI vs. Human Art dataset from [Kaggle](https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art).
Extract and place `AiArtData` and `RealArt` inside an `Art/` folder at the project root.

Expected layout:

```text
Art/
  RealArt/
  AIArtData/
```

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

## Rubric-ready workflow (EDA, tuning, model comparison)

Use these steps to satisfy the full ML project checklist (EDA + multi-model comparison + tuning + plots):

1. Extract handcrafted features:

   ```bash
   python feature_extraction.py
   ```

2. Run EDA and generate figures:

   ```bash
   python eda_analysis.py
   ```

   Outputs:
   - `reports/eda_summary.md`
   - `reports/figures/class_balance.png`
   - `reports/figures/image_size_distribution.png`
   - `reports/figures/feature_variance_hist.png`
   - `reports/figures/feature_correlation_heatmap.png`
   - `reports/figures/pca_2d.png`
   - `reports/figures/tsne_2d.png`

3. Train the CNN (needed if you want CNN in the comparison table):

   ```bash
   python train_cnn.py
   ```

4. Run hyperparameter tuning + compare **sklearn SVM**, **custom SVM** (`svm.py`, fixed hyperparameters in `tune_and_compare_models.py`), **Random Forest**, and **CNN**:

   ```bash
   python tune_and_compare_models.py
   ```

   Sklearn SVM and Random Forest use `GridSearchCV` on the handcrafted features. The custom SVM uses fixed hyperparameters defined next to `CUSTOM_SVM_*` in `tune_and_compare_models.py`. The CNN is evaluated on the **same** stratified test images (no grid search here; hyperparameters come from `train_cnn.py`).

   Outputs:
   - `reports/tuning_summary.md`
   - `reports/model_comparison.csv`
   - confusion matrices and ROC curves under `reports/figures/`
   - `saved_models/svm_best.joblib`, `saved_models/random_forest_best.joblib`, and `saved_models/custom_svm_best.joblib`

5. For your report, include:
   - dataset characteristics (size/class balance)
   - EDA findings (feature/correlation/PCA/t-SNE plots)
   - tuned hyperparameters and rationale
   - comparison across model families (linear/tree/neural)
   - metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
   - limitations and future work

## Troubleshooting

- **`FileNotFoundError` for `saved_models/ai_art_classifier_resnet.keras`:** Run `train_cnn.py` first (rubric workflow step 3). If you run `tune_and_compare_models.py` without it, CNN is skipped; sklearn SVM, custom SVM, and Random Forest are still compared.
- **Empty or wrong classes:** Ensure `Art/` has exactly the two class folders with images inside them.
- **TensorFlow install issues:** See the [official TensorFlow install guide](https://www.tensorflow.org/install) for your OS and Python version. On Windows, use Python 3.10-3.12 for best compatibility.