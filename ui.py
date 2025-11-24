import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                             QSlider, QPushButton, QFileDialog, QGridLayout, QGroupBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from project import apply_restoration_chain

class ImageUI(QWidget):
    def __init__(self):
        super().__init__()
        
        # Default Parameters
        self.blur_len = 10
        self.blur_angle = 30
        self.wiener_k = 0.005
        
        self.original_image = None
        self.processed_image = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle("DSP: Wiener Filter Real-Time Preview")
        self.setGeometry(100, 100, 1000, 700)

        # --- Layouts ---
        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        controls_layout = QGridLayout()

        # --- Image Displays ---
        self.lbl_input = QLabel("Load an Image")
        self.lbl_input.setAlignment(Qt.AlignCenter)
        self.lbl_input.setStyleSheet("border: 1px solid #444; background-color: #222; color: white;")
        self.lbl_input.setMinimumSize(400, 400)

        self.lbl_output = QLabel("Restored Result")
        self.lbl_output.setAlignment(Qt.AlignCenter)
        self.lbl_output.setStyleSheet("border: 1px solid #444; background-color: #222; color: white;")
        self.lbl_output.setMinimumSize(400, 400)

        image_layout.addWidget(self.lbl_input)
        image_layout.addWidget(self.lbl_output)

        # --- Controls ---

        # 1. Buttons Layout
        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_load.setStyleSheet("padding: 10px; font-weight: bold;")

        # --- NEW SAVE BUTTON ---
        self.btn_save = QPushButton("Save Result")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setStyleSheet("padding: 10px; font-weight: bold; background-color: #4CAF50; color: white;")
        self.btn_save.setEnabled(False) # Disable until we have an image
        # -----------------------

        # Update Grid Layout to hold both buttons
        controls_layout.addWidget(self.btn_load, 0, 0)
        controls_layout.addWidget(self.btn_save, 0, 1, 1, 2) # Span across 2 columns
        # 2. Sliders
        # Blur Length (1 to 50)
        self.sl_len = QSlider(Qt.Horizontal)
        self.sl_len.setRange(1, 60)
        self.sl_len.setValue(self.blur_len)
        self.sl_len.valueChanged.connect(self.update_params)
        self.lbl_len_val = QLabel(f"Length: {self.blur_len}")

        # Blur Angle (0 to 180)
        self.sl_angle = QSlider(Qt.Horizontal)
        self.sl_angle.setRange(0, 180)
        self.sl_angle.setValue(self.blur_angle)
        self.sl_angle.valueChanged.connect(self.update_params)
        self.lbl_angle_val = QLabel(f"Angle: {self.blur_angle}°")

        # Wiener K (scaled: 0 to 100 -> 0.0 to 0.1)
        self.sl_k = QSlider(Qt.Horizontal)
        self.sl_k.setRange(1, 200) # 1 to 200
        self.sl_k.setValue(int(self.wiener_k * 2000)) # Scale factor
        self.sl_k.valueChanged.connect(self.update_params)
        self.lbl_k_val = QLabel(f"Wiener K: {self.wiener_k}")

        # Add to Grid Layout
        controls_layout.addWidget(self.btn_load, 0, 0, 1, 2)
        
        controls_layout.addWidget(QLabel("Blur Length"), 1, 0)
        controls_layout.addWidget(self.sl_len, 1, 1)
        controls_layout.addWidget(self.lbl_len_val, 1, 2)

        controls_layout.addWidget(QLabel("Blur Angle"), 2, 0)
        controls_layout.addWidget(self.sl_angle, 2, 1)
        controls_layout.addWidget(self.lbl_angle_val, 2, 2)

        controls_layout.addWidget(QLabel("Wiener K (Noise)"), 3, 0)
        controls_layout.addWidget(self.sl_k, 3, 1)
        controls_layout.addWidget(self.lbl_k_val, 3, 2)

        # --- Assembly ---
        main_layout.addLayout(image_layout)
        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp)", 
                                                   options=options)
        if file_path:
            # Load via OpenCV
            img = cv2.imread(file_path)
            if img is None: return
            
            # Resize for performance if image is massive (> 800px width)
            h, w = img.shape[:2]
            if w > 800:
                scale = 800 / w
                img = cv2.resize(img, (800, int(h * scale)))

            self.original_image = img
            self.process_and_display()

    def process_and_display(self):
        if self.original_image is None: return

        # Call the DSP engine
        blurred, restored = apply_restoration_chain(
            self.original_image, 
            self.blur_len, 
            self.blur_angle, 
            self.wiener_k
        )

        # --- STORE THE RESULT FOR SAVING ---
        self.processed_image = restored 
        self.btn_save.setEnabled(True) # Enable the save button now
        # -----------------------------------

        # Display Left (Blurred Input)
        self.display_image(blurred, self.lbl_input)
        
        # Display Right (Restored Output)
        self.display_image(restored, self.lbl_output)

    def update_params(self):
        if self.original_image is None: return

        self.blur_len = self.sl_len.value()
        self.blur_angle = self.sl_angle.value()
        # Map slider 1-200 to float 0.0005 - 0.1
        self.wiener_k = self.sl_k.value() / 2000.0 

        # Update Labels
        self.lbl_len_val.setText(f"Length: {self.blur_len}")
        self.lbl_angle_val.setText(f"Angle: {self.blur_angle}°")
        self.lbl_k_val.setText(f"Wiener K: {self.wiener_k:.4f}")

        self.process_and_display()

    def process_and_display(self):
        if self.original_image is None: return

        # Call the DSP engine
        # Returns: (Simulated Blurred Image, Restored Image)
        blurred, restored = apply_restoration_chain(
            self.original_image, 
            self.blur_len, 
            self.blur_angle, 
            self.wiener_k
        )

        # Display Left (Blurred Input)
        self.display_image(blurred, self.lbl_input)
        
        # Display Right (Restored Output)
        self.display_image(restored, self.lbl_output)

    def display_image(self, img_cv, label_widget):
        # Convert BGR (OpenCV) to RGB (Qt)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(q_img)
        label_widget.setPixmap(pixmap.scaled(
            label_widget.width(), 
            label_widget.height(), 
            Qt.KeepAspectRatio
        ))

    def save_image(self):
        if self.processed_image is None: return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Restored Image", 
            "restored_image.png", 
            "Images (*.png *.jpg *.jpeg)", 
            options=options
        )

        if file_path:
            # OpenCV writes the numpy array to disk
            # It automatically handles encoding based on the file extension you chose
            cv2.imwrite(file_path, self.processed_image)
            print(f"Saved to: {file_path}")