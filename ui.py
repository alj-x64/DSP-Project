import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
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
        self.setWindowTitle("DSP: Image Restoration & Analysis")
        self.setGeometry(100, 100, 1300, 700) # Made window wider for side panel

        # --- Main Layout (Left: Controls/Image, Right: Analysis) ---
        main_layout = QHBoxLayout() 
        
        # Left Side Container (Images + Controls)
        left_container = QWidget()
        left_layout = QVBoxLayout()
        
        # Image Display Area
        image_row = QHBoxLayout()
        
        self.lbl_input = QLabel("Original / Blurred")
        self.lbl_input.setAlignment(Qt.AlignCenter)
        self.lbl_input.setStyleSheet("border: 1px solid #444; background-color: #222; color: #AAA;")
        self.lbl_input.setFixedSize(400, 400) # Fixed size to keep layout stable

        self.lbl_output = QLabel("Restored Result")
        self.lbl_output.setAlignment(Qt.AlignCenter)
        self.lbl_output.setStyleSheet("border: 1px solid #444; background-color: #222; color: #AAA;")
        self.lbl_output.setFixedSize(400, 400)

        image_row.addWidget(self.lbl_input)
        image_row.addWidget(self.lbl_output)

        # Controls Area
        controls_group = QGroupBox("Restoration Controls")
        controls_layout = QGridLayout()

        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_load.setStyleSheet("padding: 8px;")

        self.btn_save = QPushButton("Save Result")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("padding: 8px; background-color: #4CAF50; color: white;")

        # Sliders
        self.sl_len = QSlider(Qt.Horizontal)
        self.sl_len.setRange(1, 60)
        self.sl_len.setValue(self.blur_len)
        self.sl_len.valueChanged.connect(self.update_params)
        
        self.sl_angle = QSlider(Qt.Horizontal)
        self.sl_angle.setRange(0, 180)
        self.sl_angle.setValue(self.blur_angle)
        self.sl_angle.valueChanged.connect(self.update_params)
        
        self.sl_k = QSlider(Qt.Horizontal)
        self.sl_k.setRange(1, 200) 
        self.sl_k.setValue(int(self.wiener_k * 2000)) 
        self.sl_k.valueChanged.connect(self.update_params)

        # Labels for sliders
        self.lbl_len_val = QLabel(f"Length: {self.blur_len}")
        self.lbl_angle_val = QLabel(f"Angle: {self.blur_angle}°")
        self.lbl_k_val = QLabel(f"Wiener K: {self.wiener_k}")

        controls_layout.addWidget(self.btn_load, 0, 0)
        controls_layout.addWidget(self.btn_save, 0, 1, 1, 2)
        
        controls_layout.addWidget(QLabel("Blur Length"), 1, 0)
        controls_layout.addWidget(self.sl_len, 1, 1)
        controls_layout.addWidget(self.lbl_len_val, 1, 2)

        controls_layout.addWidget(QLabel("Blur Angle"), 2, 0)
        controls_layout.addWidget(self.sl_angle, 2, 1)
        controls_layout.addWidget(self.lbl_angle_val, 2, 2)

        controls_layout.addWidget(QLabel("Wiener K"), 3, 0)
        controls_layout.addWidget(self.sl_k, 3, 1)
        controls_layout.addWidget(self.lbl_k_val, 3, 2)
        
        controls_group.setLayout(controls_layout)
        
        left_layout.addLayout(image_row)
        left_layout.addWidget(controls_group)
        left_container.setLayout(left_layout)

        # --- Right Side Container (Analysis Plots) ---
        right_container = QGroupBox("Signal Analysis")
        right_layout = QVBoxLayout()
        right_container.setFixedWidth(350) # Fixed width for the sidebar

        self.lbl_spectrum = QLabel("Spectrum")
        self.lbl_spectrum.setAlignment(Qt.AlignCenter)
        self.lbl_spectrum.setStyleSheet("border: 1px solid #555; background-color: #111;")
        self.lbl_spectrum.setFixedHeight(300)

        self.lbl_hist = QLabel("Histogram")
        self.lbl_hist.setAlignment(Qt.AlignCenter)
        self.lbl_hist.setStyleSheet("border: 1px solid #555; background-color: #111;")
        self.lbl_hist.setFixedHeight(300)

        right_layout.addWidget(QLabel("<b>Frequency Domain (Spectrum)</b>"))
        right_layout.addWidget(self.lbl_spectrum)
        right_layout.addWidget(QLabel("<b>Pixel Intensity (Histogram)</b>"))
        right_layout.addWidget(self.lbl_hist)
        right_layout.addStretch()
        
        right_container.setLayout(right_layout)

        # Combine
        main_layout.addWidget(left_container)
        main_layout.addWidget(right_container)
        self.setLayout(main_layout)

    # ... [Load, Update, Save functions are same as before] ...
    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            img = cv2.imread(file_path)
            if img is None: return
            h, w = img.shape[:2]
            if w > 800: # Resize if huge
                scale = 800 / w
                img = cv2.resize(img, (800, int(h * scale)))
            self.original_image = img
            self.process_and_display()

    def update_params(self):
        if self.original_image is None: return
        self.blur_len = self.sl_len.value()
        self.blur_angle = self.sl_angle.value()
        self.wiener_k = self.sl_k.value() / 2000.0 
        self.lbl_len_val.setText(f"Length: {self.blur_len}")
        self.lbl_angle_val.setText(f"Angle: {self.blur_angle}°")
        self.lbl_k_val.setText(f"Wiener K: {self.wiener_k:.4f}")
        self.process_and_display()

    def save_image(self):
        if self.processed_image is None: return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save", "result.png", "Images (*.png *.jpg)", options=options)
        if file_path:
            cv2.imwrite(file_path, self.processed_image)

    def process_and_display(self):
        if self.original_image is None: return

        blurred, restored = apply_restoration_chain(
            self.original_image, self.blur_len, self.blur_angle, self.wiener_k
        )
        self.processed_image = restored
        self.btn_save.setEnabled(True)

        self.display_image(blurred, self.lbl_input)
        self.display_image(restored, self.lbl_output)

        # --- UPDATE PLOTS ---
        self.update_plots(restored)

    def display_image(self, img_cv, label_widget):
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label_widget.setPixmap(pixmap.scaled(label_widget.width(), label_widget.height(), Qt.KeepAspectRatio))

    # --- NEW PLOTTING FUNCTIONS ---
    def update_plots(self, img):
        """Generates plots and puts them in the sidebar labels"""
        
        # 1. Magnitude Spectrum (FFT)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        
        # Normalize to 0-255 for display
        mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        mag_img = np.uint8(mag_norm)
        # Color map for "Science" look
        mag_color = cv2.applyColorMap(mag_img, cv2.COLORMAP_INFERNO)
        self.display_image(mag_color, self.lbl_spectrum)

        # 2. RGB Histogram (Using Matplotlib for nice axes)
        self.plot_histogram_to_label(img, self.lbl_hist)

    def plot_histogram_to_label(self, img, label):
        """Uses Matplotlib to plot hist, saves to buffer, displays in Label"""
        # Create a small figure
        fig, ax = plt.subplots(figsize=(3.5, 3), dpi=80) # Small size
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, linewidth=1)
        
        ax.set_title("Color Distribution", fontsize=10)
        ax.set_xlim([0, 256])
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f0f0f0') # Light gray background
        
        # Save to memory buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig) # Clean up memory
        
        # Load into Qt
        buf.seek(0)
        q_img = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))