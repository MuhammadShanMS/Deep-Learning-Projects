# 👕 Fashion MNIST Classifier

A deep learning image classifier built with **TensorFlow/Keras** that recognizes clothing items from grayscale images — trained on the classic **Fashion MNIST** dataset.

---

## 📌 Overview

This project trains a fully-connected (Dense) neural network to classify 28×28 pixel grayscale images into **10 clothing categories** — from T-shirts to ankle boots. It follows the same core pipeline as digit-classification MNIST, applied to a visually more complex, real-world-style dataset.

---

## 📊 Dataset

**Fashion MNIST** — 70,000 grayscale images (28×28 px):
- 🏋️ 60,000 training images
- 🧪 10,000 test images

| Label | Class 🏷️ |
|:---:|---|
| 0 | 👕 T-shirt/top |
| 1 | 👖 Trouser |
| 2 | 🧥 Pullover |
| 3 | 👗 Dress |
| 4 | 🧥 Coat |
| 5 | 👡 Sandal |
| 6 | 👔 Shirt |
| 7 | 👟 Sneaker |
| 8 | 👜 Bag |
| 9 | 👢 Ankle boot |

---

## 🧠 Model Architecture

```python
Sequential([
    Dense(512, activation="relu"),
    Dense(256, activation="relu"),
    Dense(10, activation="softmax")
])
```

| Setting | Value |
|---|---|
| ⚙️ Optimizer | Adam |
| 📉 Loss function | Sparse Categorical Crossentropy |
| 🎯 Metric | Accuracy |
| 🔁 Epochs | 35 |
| 📦 Batch size | 128 |

---

## 🛠️ Preprocessing

- 🔄 Reshaped images from `(28, 28)` → flat vector of `784` pixels (required for Dense layers)
- 📏 Normalized pixel values from `[0, 255]` → `[0, 1]` for stable, faster training

---

## 📈 Results

| Metric | Score |
|---|---|
| ✅ Training Accuracy | ~94% |
| 🎯 Test Accuracy | ~89% |

---

## 🚀 How to Run

1. Clone this repo
2. Open `Fashion_MNIST_Classifier.ipynb` in Jupyter Notebook, Google Colab, or VS Code
3. Run all cells in order ▶️

```bash
pip install tensorflow numpy
```

---

## 🧰 Tech Stack

- 🐍 Python
- 🔶 TensorFlow / Keras
- 🔢 NumPy

---

## 📄 License

📚 This project is for educational purposes, built as part of a Deep Learning course assignment.
