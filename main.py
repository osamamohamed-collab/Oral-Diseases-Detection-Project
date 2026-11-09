import sys
from pathlib import Path

import numpy as np
from PIL import Image

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QFrame, QProgressBar,
    QMessageBox, QListWidget, QListWidgetItem, QSizePolicy
)

import json
import tensorflow as tf
from custom_layers import BackbonePreprocess  # لازم يتعمل import قبل load_model


BASE_DIR = Path(__file__).resolve().parent
# بيدوّر على الموديل في الأماكن دي بالترتيب
MODEL_CANDIDATES = [
    BASE_DIR / "models" / "Oral_Diseases_ResNet50.keras",
    BASE_DIR / "models" / "oral_best_model.keras",
    BASE_DIR / "oral_best_model.keras",
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])
IMAGE_SIZE = (224, 224)  # نفس حجم التدريب في الـ notebook

CLASS_NAMES = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Mouth Ulcer",
    "Tooth Discoloration",
]

# لو class_names.json موجود جنب الموديل، يتستخدم بدل القائمة اللي فوق (نفس ترتيب التدريب)
for _p in (MODEL_PATH.parent / "class_names.json", BASE_DIR / "class_names.json"):
    if _p.exists():
        with open(_p, encoding="utf-8") as _f:
            CLASS_NAMES = json.load(_f)
        break


def load_model():
    """Load the trained Keras model. Preprocessing and augmentation are inside the model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found:\n{MODEL_PATH}\n\n"
            "Put Oral_Diseases_ResNet50.keras inside the models folder."
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"BackbonePreprocess": BackbonePreprocess},
        compile=False,
    )


def prepare_image(image_path: str) -> np.ndarray:
    """
    Match the training input:
    RGB image -> resize to 224x224 -> float32 raw values in [0, 255].
    The model does its own preprocessing (BackbonePreprocess), so NO division by 255.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)  # raw 0..255
    return np.expand_dims(array, axis=0)


class PredictionWorker(QThread):
    finished_prediction = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, model, image_path):
        super().__init__()
        self.model = model
        self.image_path = image_path

    def run(self):
        try:
            batch = prepare_image(self.image_path)
            probabilities = self.model.predict(batch, verbose=0)[0]
            self.finished_prediction.emit(probabilities)
        except Exception as exc:
            self.failed.emit(str(exc))


class OralDiseaseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oral Diseases Detection | AI Desktop App")
        self.setMinimumSize(1000, 680)
        self.model = None
        self.image_path = None
        self.worker = None

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #101827;
                color: #E8EEF7;
                font-family: Segoe UI;
                font-size: 14px;
            }
            QLabel#title {
                font-size: 27px;
                font-weight: 700;
                color: #FFFFFF;
            }
            QLabel#subtitle {
                color: #AAB8CC;
                font-size: 13px;
            }
            QFrame#panel {
                background-color: #192438;
                border: 1px solid #2B3B55;
                border-radius: 16px;
            }
            QLabel#imagePreview {
                background-color: #0D1522;
                border: 2px dashed #40516C;
                border-radius: 12px;
                color: #AAB8CC;
                min-height: 300px;
            }
            QPushButton {
                background-color: #2878D0;
                color: white;
                border: none;
                border-radius: 9px;
                padding: 12px 18px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #368BE8; }
            QPushButton:disabled {
                background-color: #46566D;
                color: #B9C3D1;
            }
            QPushButton#secondary {
                background-color: #2A3A52;
            }
            QPushButton#secondary:hover { background-color: #394D69; }
            QLabel#resultDisease {
                color: #68D5A5;
                font-size: 23px;
                font-weight: 700;
            }
            QLabel#confidence {
                color: #FFFFFF;
                font-size: 30px;
                font-weight: 700;
            }
            QListWidget {
                background-color: #111C2D;
                border: 1px solid #2B3B55;
                border-radius: 10px;
                padding: 7px;
            }
            QListWidget::item {
                padding: 9px;
                border-bottom: 1px solid #26364D;
            }
            QProgressBar {
                background-color: #26364D;
                border: none;
                border-radius: 5px;
                height: 9px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #39B98A;
                border-radius: 5px;
            }
        """)

        self._build_ui()
        self._load_model_safely()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(28, 24, 28, 22)
        outer.setSpacing(18)

        header = QVBoxLayout()
        title = QLabel("Oral Diseases Detection")
        title.setObjectName("title")
        subtitle = QLabel(
            "AI-powered image classification using your trained ResNet50 model"
        )
        subtitle.setObjectName("subtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        outer.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(18)
        outer.addLayout(content, 1)

        # Left panel: image upload and preview
        left = QFrame()
        left.setObjectName("panel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(14)

        section = QLabel("Input Image")
        section.setStyleSheet("font-size: 18px; font-weight: 700;")
        left_layout.addWidget(section)

        self.preview = QLabel("Choose an oral image to get started")
        self.preview.setObjectName("imagePreview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setWordWrap(True)
        self.preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        left_layout.addWidget(self.preview, 1)

        buttons = QHBoxLayout()
        self.upload_btn = QPushButton("Choose Image")
        self.upload_btn.clicked.connect(self.choose_image)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("secondary")
        self.clear_btn.clicked.connect(self.clear_image)
        buttons.addWidget(self.upload_btn)
        buttons.addWidget(self.clear_btn)
        left_layout.addLayout(buttons)

        self.predict_btn = QPushButton("Predict Disease")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setEnabled(False)
        left_layout.addWidget(self.predict_btn)

        self.model_status = QLabel("Loading model...")
        self.model_status.setObjectName("subtitle")
        left_layout.addWidget(self.model_status)

        content.addWidget(left, 5)

        # Right panel: prediction and class probabilities
        right = QFrame()
        right.setObjectName("panel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 18, 20, 18)
        right_layout.setSpacing(12)

        result_title = QLabel("Prediction Result")
        result_title.setStyleSheet("font-size: 18px; font-weight: 700;")
        right_layout.addWidget(result_title)

        self.result_label = QLabel("No prediction yet")
        self.result_label.setObjectName("resultDisease")
        self.result_label.setWordWrap(True)
        right_layout.addWidget(self.result_label)

        self.confidence_label = QLabel("--")
        self.confidence_label.setObjectName("confidence")
        right_layout.addWidget(self.confidence_label)

        confidence_caption = QLabel("Model confidence for the predicted class")
        confidence_caption.setObjectName("subtitle")
        right_layout.addWidget(confidence_caption)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #2B3B55;")
        right_layout.addWidget(divider)

        scores_title = QLabel("Class probabilities")
        scores_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        right_layout.addWidget(scores_title)

        self.scores_list = QListWidget()
        right_layout.addWidget(self.scores_list, 1)

        self.disclaimer = QLabel(
            "Educational prototype only. This prediction is not a medical diagnosis. "
            "Consult a qualified dentist for clinical assessment."
        )
        self.disclaimer.setWordWrap(True)
        self.disclaimer.setStyleSheet(
            "color: #F0C477; background: #302B22; padding: 10px; border-radius: 8px;"
        )
        right_layout.addWidget(self.disclaimer)

        content.addWidget(right, 4)

        footer = QLabel(
            "Model: ResNet50  |  Input: 224 × 224 RGB  |  Classes: 6"
        )
        footer.setObjectName("subtitle")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(footer)

    def _load_model_safely(self):
        QApplication.processEvents()
        try:
            self.model = load_model()
            self.model_status.setText("Model loaded successfully")
            self.model_status.setStyleSheet("color: #68D5A5;")
            self.predict_btn.setEnabled(self.image_path is not None)
        except Exception as exc:
            self.model_status.setText("Model file not loaded")
            QMessageBox.warning(
                self,
                "Model loading problem",
                str(exc)
            )

    def choose_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Oral Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not file_path:
            return

        self.image_path = file_path
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Invalid image", "Could not open this image.")
            self.image_path = None
            return

        scaled = pixmap.scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview.setPixmap(scaled)
        self.result_label.setText("Ready to predict")
        self.confidence_label.setText("--")
        self.scores_list.clear()
        self.predict_btn.setEnabled(self.model is not None)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.preview.setPixmap(
                pixmap.scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

    def predict(self):
        if self.model is None:
            QMessageBox.warning(self, "Model unavailable", "Load the .keras model first.")
            return
        if not self.image_path:
            QMessageBox.information(self, "Choose image", "Please choose an image first.")
            return

        self.predict_btn.setEnabled(False)
        self.predict_btn.setText("Predicting...")
        self.worker = PredictionWorker(self.model, self.image_path)
        self.worker.finished_prediction.connect(self.show_prediction)
        self.worker.failed.connect(self.show_prediction_error)
        self.worker.start()

    def show_prediction(self, probabilities):
        self.predict_btn.setEnabled(True)
        self.predict_btn.setText("Predict Disease")

        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100.0

        self.result_label.setText(predicted_class)
        self.confidence_label.setText(f"{confidence:.2f}%")

        self.scores_list.clear()
        sorted_indices = np.argsort(probabilities)[::-1]
        for index in sorted_indices:
            score = float(probabilities[index]) * 100.0
            item = QListWidgetItem(f"{CLASS_NAMES[index]}     {score:.2f}%")
            self.scores_list.addItem(item)

    def show_prediction_error(self, message):
        self.predict_btn.setEnabled(True)
        self.predict_btn.setText("Predict Disease")
        QMessageBox.critical(self, "Prediction error", message)

    def clear_image(self):
        self.image_path = None
        self.preview.clear()
        self.preview.setText("Choose an oral image to get started")
        self.result_label.setText("No prediction yet")
        self.confidence_label.setText("--")
        self.scores_list.clear()
        self.predict_btn.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    window = OralDiseaseApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
