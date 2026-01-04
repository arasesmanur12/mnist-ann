# MNIST Handwritten Digit Classification (ANN & CNN)

This project focuses on handwritten digit classification using the **MNIST dataset**.
Both **Artificial Neural Network (ANN)** and **Convolutional Neural Network (CNN)** approaches are implemented and compared.

---

##  Dataset: MNIST

- 60,000 training images
- 10,000 test images
- Image size: 28×28 grayscale
- Classes: 10 (digits 0–9)

---

##  ANN Model (Artificial Neural Network)

The ANN model uses flattened image data as input.

### ANN Architecture
- Input layer: 784 neurons (28×28 flattened)
- Dense layer: 128 neurons (ReLU)
- Dropout: 0.5
- Dense layer: 64 neurons (ReLU)
- Output layer: 10 neurons (Softmax)

### ANN Training Details
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 10
- Batch size: 32

ANN implementation is available in `ann.py`.

---

## CNN Model (Convolutional Neural Network)

CNN is implemented to better capture spatial features in image data.

### Why CNN?
- ANN loses spatial information when flattening images
- CNN preserves spatial structure
- CNN performs better on image classification tasks

### CNN Architecture
- Conv2D (32 filters, 3×3) + ReLU
- MaxPooling (2×2)
- Conv2D (64 filters, 3×3) + ReLU
- MaxPooling (2×2)
- Flatten
- Dense (128 neurons) + ReLU
- Dropout (0.5)
- Output Dense (10 neurons, Softmax)

### CNN Training Details
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 5
- Batch size: 32

CNN implementation is available in `cnn.py`.

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

##  How to Run

```bash
git clone https://github.com/arasesmanur12/mnist-ann.git
cd mnist-ann
pip install -r requirements.txt
python ann.py
python cnn.py
