# Cat vs Dog Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify images as either containing a **cat** or a **dog**. The dataset is processed, normalized, and passed through a CNN for binary classification. The project is built using **TensorFlow/Keras**.

---

## 🔍 Overview
This project aims to:
1. Build a robust CNN model for binary classification.
2. Preprocess the dataset using TensorFlow's utilities.
3. Improve model performance with techniques like Batch Normalization and Dropout.
4. Visualize training and validation results for better insights.

---

## 📂 Dataset
The dataset used for this project consists of images of cats and dogs, split into training and validation sets.  
- **Training Images**: Located in `/content/train`  
- **Validation Images**: Located in `/content/test`  

The dataset is loaded using `keras.utils.image_dataset_from_directory`.

---

## 🚀 Features
- **Data Loading and Preprocessing**
  - Resizing images to 256x256.
  - Normalizing pixel values to the range `[0, 1]`.
  - Applying caching and prefetching for faster data loading.
- **CNN Architecture**
  - Three convolutional layers with `ReLU` activation.
  - Batch Normalization after each convolution.
  - MaxPooling for dimensionality reduction.
  - Dense layers with Dropout to prevent overfitting.
- **Training**
  - Optimized with the **Adam** optimizer.
  - Binary Cross-Entropy loss function for classification.
  - ModelCheckpoint and EarlyStopping callbacks.
- **Performance Metrics**
  - Training and validation accuracy and loss are tracked during training.

---

## 🛠️ Model Architecture
The CNN model consists of the following layers:
1. **Conv2D + BatchNormalization + MaxPooling** (32 filters)
2. **Conv2D + BatchNormalization + MaxPooling** (64 filters)
3. **Conv2D + BatchNormalization + MaxPooling** (128 filters)
4. **Flatten + Dense** (128 neurons)
5. **Dropout** (rate = 0.3)
6. **Dense** (64 neurons)
7. **Dropout** (rate = 0.3)
8. **Dense** (1 neuron with Sigmoid activation)

---

## 🔧 Installation
To run this project, ensure you have the following installed:
- Python 3.7 or higher
- TensorFlow 2.x
- Matplotlib
- NumPy

### Clone the repository
```bash
git clone https://github.com/irehmanar/Cat-vs-Dog-Classification-using-cnn.git
cd cat-vs-dog-classification
