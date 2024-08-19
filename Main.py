import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QHBoxLayout, QTextEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# بارگذاری مدل MobileNetV2 از پیش آموزش‌دیده
model = tf.keras.applications.MobileNetV2(weights='imagenet')

class CatFinderApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Cat Finder")
        self.setGeometry(100, 100, 650, 700)

        layout = QVBoxLayout()

        # عنوان اپلیکیشن
        title_label = QLabel("تشخیص گربه در تصویر", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px;")
        layout.addWidget(title_label)

        # فریم برای دکمه‌ها
        control_layout = QHBoxLayout()

        upload_button = QPushButton("بارگذاری تصویر", self)
        upload_button.clicked.connect(self.upload_and_process_image)
        control_layout.addWidget(upload_button)

        reset_button = QPushButton("ریست", self)
        reset_button.clicked.connect(self.reset_ui)
        control_layout.addWidget(reset_button)

        exit_button = QPushButton("خروج", self)
        exit_button.clicked.connect(self.exit_app)
        control_layout.addWidget(exit_button)

        layout.addLayout(control_layout)

        # نمایش وضعیت
        self.status_label = QLabel("آماده", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # پنل برای نمایش تصویر
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # نمایش نتایج
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; color: green;")
        layout.addWidget(self.result_label)

        # فریم برای نمایش پیش‌بینی‌ها
        self.predictions_text = QTextEdit(self)
        self.predictions_text.setReadOnly(True)
        layout.addWidget(self.predictions_text)

        # فریم برای نمایش توضیحات درباره تصویر
        self.image_info_text = QTextEdit(self)
        self.image_info_text.setReadOnly(True)
        layout.addWidget(self.image_info_text)

        self.setLayout(layout)

    def process_image(self, image_path):
        image = cv2.imread(image_path)

        if image is None:
            return None, "تصویر بارگذاری نشد. لطفاً مسیر فایل را بررسی کنید."

        image_resized = cv2.resize(image, (224, 224))
        image_array = preprocess_input(image_resized)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        cat_labels = ['Egyptian_cat', 'tabby', 'tiger_cat', 'Persian_cat', 'Siamese_cat']
        is_cat_present = any(label in cat_labels for (_, label, _) in decoded_predictions)

        result = "گربه در تصویر پیدا نشد."
        if is_cat_present:
            result = "گربه در تصویر پیدا شد!"

        return result, decoded_predictions

    def upload_and_process_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "انتخاب تصویر", "", "Image Files (*.jpg *.jpeg *.png)")

        if not file_path:
            return

        self.status_label.setText("در حال پردازش تصویر...")

        result, predictions = self.process_image(file_path)

        if not result:
            QMessageBox.critical(self, "Error", predictions)
        else:
            self.result_label.setText(result)
            self.display_image(file_path)
            self.update_predictions(predictions)
            self.update_image_info(file_path, result, predictions)
            self.status_label.setText("آماده")

    def display_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((250, 250), Image.Resampling.LANCZOS)  # اصلاح این خط
        image.save("temp_image.jpg")
        pixmap = QPixmap("temp_image.jpg")
        self.image_label.setPixmap(pixmap)

    def update_predictions(self, predictions):
        predictions_text = ""
        for i, (_, label, score) in enumerate(predictions):
            predictions_text += f"{i+1}: {label} ({score * 100:.2f}%)\n"
        self.predictions_text.setText(predictions_text)

    def update_image_info(self, file_path, result, predictions):
        image = Image.open(file_path)
        image_format = image.format
        image_size = image.size

        info_text = (
            f"فرمت تصویر: {image_format}\n"
            f"اندازه تصویر: {image_size[0]}x{image_size[1]}\n\n"
            f"نتیجه تحلیل: {result}\n"
            f"پیش‌بینی‌ها:\n"
        )
        for i, (_, label, score) in enumerate(predictions):
            info_text += f"  - {label}: {score * 100:.2f}%\n"

        self.image_info_text.setText(info_text)

    def reset_ui(self):
        self.image_label.clear()
        self.result_label.clear()
        self.predictions_text.clear()
        self.image_info_text.clear()
        self.status_label.setText("آماده")

    def exit_app(self):
        QApplication.instance().quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CatFinderApp()
    ex.show()
    sys.exit(app.exec_())
