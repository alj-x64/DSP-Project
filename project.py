import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# ============================
# MOTION BLUR PSF (Needed for Wiener)
# ============================
def motion_blur_psf(length, angle):
    length = max(1, length)
    EPS = 1e-8
    psf = np.zeros((length, length))
    center = length // 2
    x = np.linspace(-center, center, length)
    y = np.linspace(-center, center, length)
    X, Y = np.meshgrid(x, y)
    angle = np.deg2rad(angle)
    psf[np.abs(X * np.cos(angle) + Y * np.sin(angle)) < 0.5] = 1
    psf /= psf.sum() + EPS
    return psf

# ============================
# WIENER FILTER + ENHANCEMENT
# ============================
def wiener_inverse_filter_color(img, psf, K=0.001):
    restored_channels = []
    for i in range(3):
        channel = img[:, :, i]
        G = np.fft.fft2(channel)
        H = np.fft.fft2(psf, s=channel.shape)
        H_conj = np.conj(H)
        F_est = (H_conj / (np.abs(H)**2 + K)) * G
        f_restored = np.abs(np.fft.ifft2(F_est))
        f_restored = cv2.normalize(f_restored, None, 0, 255, cv2.NORM_MINMAX)
        restored_channels.append(np.uint8(np.clip(f_restored, 0, 255)))

    restored = cv2.merge(restored_channels)

    # CLAHE (Contrast Enhancement)
    lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    restored = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Unsharp Masking (Sharpening)
    blurred_post = cv2.GaussianBlur(restored, (0, 0), sigmaX=1)
    restored = cv2.addWeighted(restored, 1.4, blurred_post, -0.4, 0)
    
    return np.uint8(cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX))

# ============================
# PLOTTING TOOLS
# ============================
def save_spectrum(img, filename):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap='inferno')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_histogram(img, filename):
    plt.figure(figsize=(6, 4))
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()