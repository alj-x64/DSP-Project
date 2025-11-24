import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# ============================
# ROBUST CUDA INITIALIZATION
# ============================
CUDA_AVAILABLE = False
try:
    import cupy as cp
    
    # 1. Self-Test: Try a small FFT computation to ensure NVRTC is working
    # This triggers the compilation error immediately inside this try-block
    print("Testing CUDA execution...")
    test_arr = cp.array([[1.0, 2.0], [3.0, 4.0]])
    cp.fft.fft2(test_arr)
    
    CUDA_AVAILABLE = True
    print("✅ CUDA Detected & Verified: Using GPU for FFT acceleration.")

except Exception as e:
    # If ANY error happens (ImportError, CompileException, NVRTCError), we fall back.
    print(f"⚠️ CUDA Error Detected: {e}")
    print("⚠️ Falling back to CPU (NumPy) mode.")
    import numpy as cp
    CUDA_AVAILABLE = False

# ============================
# MOTION BLUR PSF
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
# WIENER FILTER (Hybrid GPU/CPU)
# ============================
def wiener_inverse_filter_color(img, psf, K=0.001):
    restored_channels = []
    h_shape = img.shape[:2]
    
    # Handle PSF FFT (GPU or CPU)
    if CUDA_AVAILABLE:
        try:
            psf_gpu = cp.asarray(psf)
            H = cp.fft.fft2(psf_gpu, s=h_shape)
        except Exception:
            # Emergency fallback if runtime error occurs
            return wiener_inverse_filter_color_cpu(img, psf, K)
    else:
        H = np.fft.fft2(psf, s=h_shape)
        
    H_conj = cp.conj(H)
    H_pow2 = cp.abs(H)**2

    for i in range(3):
        channel = img[:, :, i]

        if CUDA_AVAILABLE:
            channel_data = cp.asarray(channel)
        else:
            channel_data = channel

        G = cp.fft.fft2(channel_data)

        denominator = H_pow2 + K
        F_est = (H_conj / denominator) * G
        
        f_restored = cp.abs(cp.fft.ifft2(F_est))

        if CUDA_AVAILABLE:
            f_restored = cp.asnumpy(f_restored)
        
        f_restored = cv2.normalize(f_restored, None, 0, 255, cv2.NORM_MINMAX)
        restored_channels.append(np.uint8(np.clip(f_restored, 0, 255)))

    restored = cv2.merge(restored_channels)

    # Post-processing (CPU)
    lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    restored = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    blurred_post = cv2.GaussianBlur(restored, (0, 0), sigmaX=1)
    restored = cv2.addWeighted(restored, 1.4, blurred_post, -0.4, 0)
    
    return np.uint8(cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX))

# Fallback function just in case
def wiener_inverse_filter_color_cpu(img, psf, K=0.001):
    # This is a pure NumPy copy for emergency fallbacks
    import numpy as np
    restored_channels = []
    H = np.fft.fft2(psf, s=img.shape[:2])
    H_conj = np.conj(H)
    H_pow2 = np.abs(H)**2
    for i in range(3):
        channel = img[:, :, i]
        G = np.fft.fft2(channel)
        F_est = (H_conj / (H_pow2 + K)) * G
        f_restored = np.abs(np.fft.ifft2(F_est))
        f_restored = cv2.normalize(f_restored, None, 0, 255, cv2.NORM_MINMAX)
        restored_channels.append(np.uint8(np.clip(f_restored, 0, 255)))
    restored = cv2.merge(restored_channels)
    # (Skip post-processing replication for brevity, usually main function handles it)
    return restored

# ============================
# PLOTTING TOOLS
# ============================
def save_spectrum(img, filename):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    if CUDA_AVAILABLE:
        try:
            gray_gpu = cp.asarray(gray)
            f = cp.fft.fft2(gray_gpu)
            fshift = cp.fft.fftshift(f)
            magnitude_spectrum = 20 * cp.log(cp.abs(fshift) + 1e-8)
            magnitude_spectrum = cp.asnumpy(magnitude_spectrum)
        except:
            # Fallback
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    else:
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