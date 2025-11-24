import cv2
import numpy as np
import os
from pathlib import Path

# ============================
# MOTION BLUR PSF CREATION
# ============================
def motion_blur_psf(length, angle):
    """Generate a motion blur Point Spread Function (PSF)."""
    # Safety check for length
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
# WIENER FILTER RESTORATION
# ============================
def wiener_inverse_filter_color(img, psf, K=0.001):
    """Apply Wiener filter channel-wise on color image."""
    restored_channels = []
    # Process B, G, R individually
    for i in range(3):
        channel = img[:, :, i]
        G = np.fft.fft2(channel)
        H = np.fft.fft2(psf, s=channel.shape)
        H_conj = np.conj(H)
        F_est = (H_conj / (np.abs(H)**2 + K)) * G
        f_restored = np.abs(np.fft.ifft2(F_est))
        
        # Normalize and clip
        f_restored = cv2.normalize(f_restored, None, 0, 255, cv2.NORM_MINMAX)
        restored_channels.append(np.uint8(np.clip(f_restored, 0, 255)))

    restored = cv2.merge(restored_channels)

    # Contrast enhancement (CLAHE) + Unsharp Masking
    lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    restored = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    blurred_post = cv2.GaussianBlur(restored, (0, 0), sigmaX=1)
    restored = cv2.addWeighted(restored, 1.4, blurred_post, -0.4, 0)

    restored = cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(restored)

# ============================
# WRAPPER FOR UI
# ============================
def apply_restoration_chain(img, length, angle, k_val):
    """
    1. Generates PSF based on inputs.
    2. Simulates Blur (so you can see what you are fixing).
    3. Restores using Wiener.
    Returns: (Blurred Image, Restored Image)
    """
    psf = motion_blur_psf(length, angle)
    
    # Restore
    restored = wiener_inverse_filter_color(img, psf, k_val)
    
    return restored

# ============================
# BATCH PROCESSING (Legacy)
# ============================
if __name__ == "__main__":
    # Only run this if executing project.py directly
    INPUT_FOLDER = Path(r"data")
    OUTPUT_FOLDER = Path(r"results")
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    LEN = 20
    THETA = 45
    K = 0.01

    images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png'))]
    for img_file in images:
        path = os.path.join(INPUT_FOLDER, img_file)
        img = cv2.imread(path)
        if img is not None:
            _, restored = apply_restoration_chain(img, LEN, THETA, K)
            cv2.imwrite(str(OUTPUT_FOLDER / f"restored_{img_file}"), restored)
            print(f"Processed {img_file}")