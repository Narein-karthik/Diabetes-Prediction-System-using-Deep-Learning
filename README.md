# Diabetes Prediction with Keras (CUDA Optimized)

An Artificial Neural Network (ANN) implementation using Keras and TensorFlow to predict the onset of diabetes. This project is optimized to leverage NVIDIA GPU acceleration for faster training.

## 🚀 Features
- **GPU Acceleration:** Configured to run on NVIDIA RTX 40-series GPUs using CUDA.
- **Dynamic Memory Growth:** Prevents TensorFlow from allocating all VRAM at once.
- **Binary Classification:** Built using a 3-layer Dense architecture.

## 🛠️ Hardware & Software
- **GPU:** NVIDIA GeForce RTX 4060 (8GB VRAM)
- **Environment:** PyCharm / Python 3.9+
- **Libraries:** TensorFlow, NumPy

## 📋 How to Use
1. Clone this repository or download the files.
2. Ensure `pima-indians-diabetes.csv` is in the same directory.
3. Install dependencies:
   ```bash
   pip install tensorflow numpy
