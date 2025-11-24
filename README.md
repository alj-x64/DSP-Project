# DSP Image Processing Project Guide
## Color Image Restoration Using Wiener Filtering

### **Group Information**
- Cascante, Betty Mae – 22-07751
- Gorreon, James Andrew – 
- Dela Cruz, Angelo – 
- Catiban, Aljon – 21-11145

---

## **Project Title**
**Color Image Restoration Using Wiener Filtering in the Frequency Domain**

---

## **Objective**
This project aims to restore motion-blurred color images using frequency-domain Wiener filtering.

---

## **DSP Concept**

### **1. Motion Blur PSF**
Motion blur is modeled using a linear PSF determined by:
- **Length (LEN)**
- **Angle (THETA)**

### **2. Wiener Inverse Filter**
The Wiener filter reconstructs the original image in the frequency domain.

---

## **Process Overview**
1. Load images by clicking "Load Image" button then select your image to be processed
2. Generate PSF using LEN and THETA
3. Simulate motion blur
4. Restore using Wiener filter (per channel)
5. Apply CLAHE for contrast improvement
6. Enhance sharpness with unsharp masking
7. Compute MSE and SSIM metrics
8. Save results by clicking the save button.

---

## **Files Included**
- **main.py** – Main Python script
- **ui.py** - Script that handles the UI
- **project.py** - Script that handles restoration
- **README.md** – Documentation  

---

## **Expected Output**
- Motion-blurred image restored using Wiener filter  
- Enhanced contrast and sharpness  
- Cleaned and visually improved version of the image  
- MSE and SSIM printed in console  
- Restored images saved in location you specify via a dialog

---

## **Metrics Used**

### **Mean Squared Error (MSE)**
Measures pixel difference between original and restored image.

### **Structural Similarity Index (SSIM)**
Measures visual similarity (0–1 scale), with 1 being identical.

---

## **How to Run**
In Linux,
```bash
python main.py
```

In Windows,
```
python main.py
```
