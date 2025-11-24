import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                             QSlider, QPushButton, QFileDialog, QGridLayout, QGroupBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from project import wiener_inverse_filter_color, motion_blur_psf, save_spectrum, save_histogram

class ImageUI(QWidget):
    def __init__(self):
        super().__init__()
        
        # Default Parameters
        self.blur_len = 10
        self.blur_angle = 30
        self.wiener_k = 0.005
        
        self.original_image = None  # The Clean Reference
        self.processed_image = None # The Restored Output

        self.initUI()

    def initUI(self):
        self.setWindowTitle("DSP: Image Restoration with Metrics (MSE/SSIM)")
        self.setGeometry(50, 50, 1400, 750) 

        main_layout = QHBoxLayout() 
        
        # --- Left Side ---
        left_container = QWidget()
        left_layout = QVBoxLayout()
        
        # Image Display
        image_row = QHBoxLayout()
        
        # LEFT: CLEAN REFERENCE
        self.lbl_input = QLabel("Original (Clean Reference)")
        self.lbl_input.setAlignment(Qt.AlignCenter)
        self.lbl_input.setStyleSheet("border: 2px solid #4CAF50; background-color: #222; color: #AAA;")
        self.lbl_input.setFixedSize(450, 450) 

        # RIGHT: RESTORED OUTPUT
        self.lbl_output = QLabel("Restored Output")
        self.lbl_output.setAlignment(Qt.AlignCenter)
        self.lbl_output.setStyleSheet("border: 2px solid #2196F3; background-color: #222; color: #AAA;")
        self.lbl_output.setFixedSize(450, 450)

        image_row.addWidget(self.lbl_input)
        image_row.addWidget(self.lbl_output)

        # Controls
        controls_group = QGroupBox("Simulation & Restoration Controls")
        controls_layout = QGridLayout()

        self.btn_load = QPushButton("Load CLEAN Image")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_load.setStyleSheet("padding: 10px; font-weight: bold;")

        self.btn_save = QPushButton("Save Result")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("padding: 10px; background-color: #4CAF50; color: white; font-weight: bold;")

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

        self.lbl_len_val = QLabel(f"Length: {self.blur_len}")
        self.lbl_angle_val = QLabel(f"Angle: {self.blur_angle}°")
        self.lbl_k_val = QLabel(f"Wiener K: {self.wiener_k}")

        controls_layout.addWidget(self.btn_load, 0, 0)
        controls_layout.addWidget(self.btn_save, 0, 1)
        
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

        # --- Right Side (Analysis & Metrics) ---
        right_container = QGroupBox("Analysis")
        right_layout = QVBoxLayout()
        right_container.setFixedWidth(320)

        # METRICS BOX
        metrics_box = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout()
        self.lbl_mse = QLabel("MSE: 0.00")
        self.lbl_mse.setStyleSheet("font-size: 16px; font-weight: bold; color: #0047AB;") # Yellow
        self.lbl_ssim = QLabel("SSIM: 1.00")
        self.lbl_ssim.setStyleSheet("font-size: 16px; font-weight: bold; color: #00BCD4;") # Cyan
        metrics_layout.addWidget(self.lbl_mse)
        metrics_layout.addWidget(self.lbl_ssim)
        metrics_box.setLayout(metrics_layout)

        # PLOTS
        self.lbl_spectrum = QLabel()
        self.lbl_spectrum.setAlignment(Qt.AlignCenter)
        self.lbl_spectrum.setStyleSheet("border: 1px solid #555; background-color: black;")
        self.lbl_spectrum.setFixedHeight(250)

        self.lbl_hist = QLabel()
        self.lbl_hist.setAlignment(Qt.AlignCenter)
        self.lbl_hist.setStyleSheet("border: 1px solid #555; background-color: #f0f0f0;")
        self.lbl_hist.setFixedHeight(250)

        right_layout.addWidget(metrics_box)
        right_layout.addWidget(QLabel("<b>Spectrum (Restored)</b>"))
        right_layout.addWidget(self.lbl_spectrum)
        right_layout.addWidget(QLabel("<b>Histogram (Restored)</b>"))
        right_layout.addWidget(self.lbl_hist)
        right_layout.addStretch()
        
        right_container.setLayout(right_layout)

        main_layout.addWidget(left_container)
        main_layout.addWidget(right_container)
        self.setLayout(main_layout)

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CLEAN Image", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            img = cv2.imread(file_path)
            if img is None: return
            h, w = img.shape[:2]
            if w > 800: 
                scale = 800 / w
                img = cv2.resize(img, (800, int(h * scale)))
            
            self.original_image = img
            # Show Original on Left
            self.display_image(self.original_image, self.lbl_input)
            self.process_image()

    def update_params(self):
        if self.original_image is None: return
        self.blur_len = self.sl_len.value()
        self.blur_angle = self.sl_angle.value()
        self.wiener_k = self.sl_k.value() / 2000.0 
        self.lbl_len_val.setText(f"Length: {self.blur_len}")
        self.lbl_angle_val.setText(f"Angle: {self.blur_angle}°")
        self.lbl_k_val.setText(f"Wiener K: {self.wiener_k:.4f}")
        self.process_image()

    def process_image(self):
        # 1. Generate PSF
        psf = motion_blur_psf(self.blur_len, self.blur_angle)
        
        # 2. SIMULATE BLUR (Internally)
        # We blur the clean image so we have something to fix
        degraded = cv2.filter2D(self.original_image, -1, psf)
        
        # 3. RESTORE
        self.processed_image = wiener_inverse_filter_color(degraded, psf, self.wiener_k)
        self.btn_save.setEnabled(True)

        # 4. Show Result on Right
        self.display_image(self.processed_image, self.lbl_output)

        # 5. CALCULATE METRICS (Clean vs Restored)
        gray_orig = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        gray_rest = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        
        m_val = mse(gray_orig, gray_rest)
        s_val = ssim(gray_orig, gray_rest)
        
        self.lbl_mse.setText(f"MSE: {m_val:.2f}")
        self.lbl_ssim.setText(f"SSIM: {s_val:.4f}")

        # 6. Update Plots
        self.update_plots(self.processed_image)

    def save_image(self):
        if self.processed_image is None: return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Result", "restored.png", "Images (*.png *.jpg)", options=options)
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            base_name = file_path.rsplit('.', 1)[0]
            save_spectrum(self.processed_image, f"{base_name}_spectrum.png")
            save_histogram(self.processed_image, f"{base_name}_histogram.png")

    def display_image(self, img_cv, label_widget):
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label_widget.setPixmap(pixmap.scaled(label_widget.width(), label_widget.height(), Qt.KeepAspectRatio))

    def update_plots(self, img):
        # Spectrum
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        mag_color = cv2.applyColorMap(np.uint8(mag_norm), cv2.COLORMAP_INFERNO)
        self.display_image(mag_color, self.lbl_spectrum)

        # Histogram
        fig, ax = plt.subplots(figsize=(3.5, 3), dpi=80)
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, linewidth=1)
        ax.set_title("Color Distribution", fontsize=10)
        ax.set_xlim([0, 256])
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f0f0f0')
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        
        buf.seek(0)
        q_img = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(q_img)
        self.lbl_hist.setPixmap(pixmap.scaled(self.lbl_hist.width(), self.lbl_hist.height(), Qt.KeepAspectRatio))