# Deep Learning Projects

A collection of deep learning projects, models and implementations built with Python, TensorFlow and Keras.

## Projects

| # | Project | Type | Test Accuracy |
|---|---------|------|---------------|
| 1 | [Face Mask Detection (ANN)](#1-face-mask-detection-using-ann) | Binary image classification | 94.03% |

---

## 1. Face Mask Detection using ANN

A binary image classifier that predicts whether a person is **wearing a face mask or not**, built with an Artificial Neural Network (ANN) in TensorFlow/Keras.

**Notebook:** `Face_Mask_Detection.ipynb`

### Highlights
- Achieves **94.03% test accuracy** with a fully connected network (no convolutional layers).
- Uses data augmentation, Batch Normalization, Dropout, early stopping and learning-rate scheduling to improve generalization.
- Complete evaluation: accuracy/loss curves, confusion matrix, precision/recall/F1 and sample predictions.

### Dataset
- **Source:** Kaggle, *Face Mask Dataset (With and Without Mask)* (`belsonraja/face-mask-dataset-with-and-without-mask`)
- **Size:** 3,833 images (1,915 `with_mask`, 1,918 `without_mask`)
- **Split (70/20/10):**

| Set | Images |
|-----|--------|
| Train | 2,682 |
| Validation | 766 |
| Test | 385 |

Class mapping: `with_mask = 0`, `without_mask = 1`.

### Methodology
1. **Download and split:** dataset downloaded with `kagglehub` and split with `split-folders` (seed 42).
2. **Preprocessing:** images resized to 64x64 and rescaled to the 0-1 range.
3. **Data augmentation (training set only):** rotation (15 degrees), width/height shift (10%), zoom (15%), brightness (0.8-1.2) and horizontal flip. Validation and test sets are only rescaled.
4. **Model:** fully connected ANN (see below).
5. **Training:** Adam optimizer (learning rate 0.001), binary cross-entropy loss, batch size 32, up to 40 epochs.
6. **Callbacks:**
   - `EarlyStopping` on validation loss (patience 7, best weights restored)
   - `ReduceLROnPlateau` (factor 0.5, patience 3)
7. **Evaluation:** performed on the held-out test set that was never used during training.

### Model Architecture

| Layer | Output Shape | Notes |
|-------|--------------|-------|
| Input | 64 x 64 x 3 | RGB image |
| Flatten | 12,288 | |
| Dense + BatchNorm + ReLU + Dropout(0.4) | 256 | |
| Dense + BatchNorm + ReLU + Dropout(0.3) | 128 | |
| Dense + BatchNorm + ReLU + Dropout(0.2) | 64 | |
| Dense (sigmoid) | 1 | Output probability |

Total parameters: **3,188,993** (3,188,097 trainable).

### Results (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | **94.03%** |
| Loss | **0.1428** |

**Classification report**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| with_mask | 0.92 | 0.96 | 0.94 | 192 |
| without_mask | 0.96 | 0.92 | 0.94 | 193 |

**Confusion matrix**

| | Predicted with_mask | Predicted without_mask |
|---|---|---|
| **Actual with_mask** | 184 | 8 |
| **Actual without_mask** | 15 | 178 |

The best epoch (36) was restored by early stopping. Training and validation curves are stable, with no signs of overfitting.

### How the Prediction Works
The network outputs a probability between 0 and 1:
- below 0.5 means **With Mask**
- 0.5 or above means **Without Mask**

The further the value is from 0.5, the more confident the model is.

### Limitations
- An ANN flattens the image and cannot capture spatial features as well as a CNN, which limits the achievable accuracy.
- The model is trained on close-up face crops, so it may perform poorly on full-scene photos with several people or cluttered backgrounds.
- Faces with sunglasses can occasionally be confused with masked faces.

### Future Work
- Replace the ANN with a CNN or use transfer learning (for example MobileNetV2) for higher accuracy.
- Add a face detector so the model works on full images.
- Deploy as a Streamlit web app for real-time predictions.

### How to Run
1. Open `Face_Mask_Detection.ipynb` in Google Colab and select a GPU runtime.
2. Run all cells from top to bottom. The dataset is downloaded automatically from Kaggle.

### Tech Stack
Python, TensorFlow/Keras, NumPy, Matplotlib, scikit-learn, kagglehub, split-folders

---

## Author
**Muhammad Shan**
GitHub: [MuhammadShanMS](https://github.com/MuhammadShanMS)
