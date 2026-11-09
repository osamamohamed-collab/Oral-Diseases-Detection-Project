# Oral-Diseases-Detection-Project

An end-to-end deep learning project that classifies intraoral photographs into **six oral conditions**. It covers a leakage-safe data pipeline, a custom CNN built from scratch, three fine-tuned pretrained backbones, a validation-based model comparison, and a **PyQt desktop application** for inference.

> **Medical disclaimer:** This is an educational prototype. Its predictions are **not** a medical diagnosis. Always consult a qualified dentist for clinical assessment.

<p align="center">
  <img src="docs/screenshot.png" alt="Desktop app screenshot" width="800">
</p>

---

## Table of Contents
- [Classes](#classes)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Desktop App](#running-the-desktop-app)
- [Training Your Own Model](#training-your-own-model)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Classes

| # | Class | Description |
|---|-------|-------------|
| 1 | Calculus | Hardened dental plaque (tartar) on the teeth |
| 2 | Caries | Tooth decay / cavities |
| 3 | Gingivitis | Inflammation of the gums |
| 4 | Hypodontia | Congenital absence of one or more teeth |
| 5 | Mouth Ulcer | Ulcerative lesions in the oral mucosa |
| 6 | Tooth Discoloration | Stained or discolored teeth |

## Dataset

The project uses the public Kaggle dataset **[Oral Diseases](https://www.kaggle.com/datasets/salmansajid05/oral-diseases)** (`salmansajid05/oral-diseases`).

Only the **original** image folders are used. The pre-augmented and preview folders shipped with the dataset are offline copies and are intentionally ignored, because they would leak near-identical images across the train/validation/test splits.

> The dataset is **not** included in this repository. Please download it from Kaggle.

## Methodology

### 1. Data preparation
- Labels are derived from the correct class folder.
- **Duplicate removal:** exact duplicates (MD5 hash) and near-duplicates, including mirrored copies.
- **Group-aware splitting** with `StratifiedGroupKFold`, so an image and its copies never end up in different splits (no train/test leakage).
- Class imbalance is handled with class weights.

### 2. Models
| Model | Type | Notes |
|-------|------|-------|
| Custom CNN | From scratch | 5 conv blocks (2x Conv + BatchNorm + ReLU, MaxPool, Dropout), Global Average Pooling, Dense(256) head |
| EfficientNetB0 | Transfer learning | ImageNet weights |
| ResNet50 | Transfer learning | ImageNet weights |
| DenseNet121 | Transfer learning | ImageNet weights |
| MobileNetV2 | Transfer learning | ImageNet weights, lightweight model suited for low-resource devices |

All pretrained models share the same pipeline: `Input -> Augmentation -> Backbone-specific preprocessing -> Backbone -> GAP -> Dropout -> Dense(256) -> Dropout -> Softmax`.
Augmentation (flip, rotation, translation, zoom, contrast, brightness) and preprocessing are **part of the saved model**, so inference only needs a raw RGB image.

### 3. Training strategy
- **Stage 1:** train the classification head with the backbone frozen.
- **Stage 2:** fine-tune the top 30% of the backbone (BatchNorm layers stay frozen) with a small learning rate (`2e-5`).
- Callbacks monitor **macro-F1** rather than validation loss/accuracy, since the classes are imbalanced.

### 4. Evaluation
- Model selection is done on the **validation** set; the **test** set is opened only once at the end.
- Metrics: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, and flip test-time augmentation (TTA).

## Results

> Fill in the numbers from the final cells of the notebook.

**Validation (used for model selection)**

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Custom CNN | _TBD_ | _TBD_ |
| EfficientNetB0 | _TBD_ | _TBD_ |
| ResNet50 | _TBD_ | _TBD_ |
| DenseNet121 | _TBD_ | _TBD_ |
| MobileNetV2 | _TBD_ | _TBD_ |

**Test (final, reported once)**

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Custom CNN | _TBD_ | _TBD_ |
| EfficientNetB0 | _TBD_ | _TBD_ |
| ResNet50 | _TBD_ | _TBD_ |
| DenseNet121 | _TBD_ | _TBD_ |
| MobileNetV2 | _TBD_ | _TBD_ |

The model deployed in the desktop app is **ResNet50**.

## Project Structure

```
Oral-Diseases-Detection-Project/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── screenshot.png
├── notebook/
│   └── oral_diseases_improved.ipynb   # data prep, training, comparison (Kaggle)
└── Oral_App/
    ├── main.py                        # PyQt6 desktop application
    ├── custom_layers.py               # BackbonePreprocess custom Keras layer
    ├── class_names.json               # class order used during training
    └── models/
        └── Oral_Diseases_ResNet50.keras   # trained model (see below)
```

## Installation

**Requirements:** Python **3.10 - 3.12** is recommended (TensorFlow may not yet support the newest Python releases).

```bash
git clone https://github.com/<your-username>/Oral-Diseases-Detection-Project.git
cd Oral-Diseases-Detection-Project

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install tensorflow pillow numpy PyQt6
```

### Download the trained model
The `.keras` model file is too large for a regular Git commit. Download it from the **[Releases](../../releases)** page and place it at:

```
Oral_App/models/Oral_Diseases_ResNet50.keras
```

Also make sure `Oral_App/class_names.json` is present.

## Running the Desktop App

```bash
cd Oral_App
python main.py          # or: py main.py on Windows
```

1. Wait for **"Model loaded successfully"**.
2. Click **Choose Image** and pick a `.png`, `.jpg`, `.jpeg`, `.bmp`, or `.webp` file.
3. Click **Predict Disease** to see the predicted class, its confidence, and the probability of every class.

### Troubleshooting
| Problem | Fix |
|---------|-----|
| `Could not locate class 'BackbonePreprocess'` | Keep `custom_layers.py` next to `main.py`; it must be imported before the model is loaded |
| Wrong or random predictions | Do **not** divide the image by 255; the model preprocesses raw `0-255` RGB input itself |
| Input shape error | The model expects `224 x 224` images |
| `No module named ...` | Install the missing package inside the active environment |

## Training Your Own Model

Open `notebook/oral_diseases_improved.ipynb` on Kaggle (add the dataset above as input) and run all cells. The notebook saves `oral_best_model.keras` and `class_names.json` to `/kaggle/working/models`. Copy both files to `Oral_App/models/`.

> The app, the model, and `custom_layers.py` must be consistent: the custom layer is registered under the package name `oral`, and the model must be loaded with `BackbonePreprocess` available.

## Limitations

- The model predicts **one** class per image, but real photos can show several conditions at once (e.g., calculus together with gingivitis).
- Some classes (Caries, Tooth Discoloration, Gingivitis) have relatively few images, so their metrics are less stable.
- Performance on photos from other cameras, lighting conditions, or with watermarks may be lower than on the test set.
- Not validated clinically and not intended for real-world diagnosis.

## Future Work

- Multi-label classification for images with more than one condition.
- Collecting more data for the under-represented classes.
- Larger input resolution and newer backbones (e.g., EfficientNetV2, ConvNeXt).
- Explainability with Grad-CAM heatmaps in the desktop app.
- Packaging the app as a standalone executable.

## License

Add your license here (e.g., MIT).

## Acknowledgements

- Dataset: *Oral Diseases* by `salmansajid05` on Kaggle.
- Built with TensorFlow / Keras and PyQt6.
