"""
MNIST veri seti:
rakamlama: 0-9 toplam 10 sınıf var
28x28 boyutunda gri tonlamalı görüntülerden oluşur.
Eğitim seti: 60.000 görüntü içerir.
Test seti: 10.000 görüntü içerir.       
grayscale: gri tonlamalı    
amacımız: el yazısı rakamları doğru bir şekilde sınıflandırmak

Image preprocessing:
histogram equalization:kontrast artırma
gaussian blur:gürültü azaltma
canny edge detection:kenar tespiti

ANN ile  MNIST sınıflandırması:

libraries:
tensorflow:keras ile ann modeli oluşturma ve geliştirme
matplotlib:görselleştirme
cv2:opencv image proccesing 

"""

# Importing necessary libraries
import cv2 #opencv
import numpy as np #numerical operations
import matplotlib.pyplot as plt #visualization
from tensorflow.keras.datasets import mnist #mnist dataset
from tensorflow.keras.models import Sequential #ann model
from tensorflow.keras.layers import Dense,  Dropout, Flatten  #ann layers
from tensorflow.keras.optimizers import Adam #optimizer



#load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()    
print("Training data shape:", x_train.shape, y_train.shape)
print("Testing data shape:", x_test.shape, y_test.shape)

"""
x_train: (60000, 28, 28) -> 60000 adet 28x28 boyutunda görüntü
y_train: (60000,) -> 60000 adet etiket (0-9 arası rakamlar)
x_test: (10000, 28, 28) -> 10000 adet 28x28 boyutunda görüntü
y_test: (10000,) -> 10000 adet etiket (0-9 arası rakamlar)

"""
#image preprocessing

img=x_train[5] #ilk resmi al.
stages={"original image":img} #orjinal resmi stages sözlüğüne ekle


#histogram equalization
img_eq=cv2.equalizeHist(img)
stages["histogram equalization"]=img_eq

#gaussian blur
img_blur=cv2.GaussianBlur(img_eq,(5,5),0)
stages["gaussian blur"]=img_blur

#canny edge detection
img_canny=cv2.Canny(img_blur,50,150) #kenar tespiti
stages["canny edge detection"]=img_canny


#visualize preprocessing stages
fig, axes = plt.subplots(2, 2, figsize=(6, 6)) #2*2 grid
axes=axes.flat
for ax ,( title, image) in zip(axes, stages.items()):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.suptitle("MNIST Preprocessing Stages", fontsize=16)
plt.tight_layout()
plt.show()

#preprocessing function
def preprocess_image(img):
    img_eq = cv2.equalizeHist(img)
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)
    img_canny = cv2.Canny(img_blur, 50, 150)
    features = img_canny.flatten() / 255.0  # (784,)
    return features


num_train = 60000
num_test = 10000

X_train=np.array([preprocess_image(img) for img in x_train[:num_train]])
y_train_subset=y_train[:num_train]

X_test=np.array([preprocess_image(img) for img in x_test[:num_test]])
y_test_subset=y_test[:num_test]


#ann model creation

model=Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,))) #ilk katman,128*28=784
model.add(Dropout(0.5)) #overfitting önleme
model.add(Dense(64, activation='relu')) #ikinci katman 64 nöron
model.add(Dense(10, activation='softmax')) #çıkış katmanı 10 nöron


#compile model 
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary()) 


#ann model training

history=model.fit(
    X_train, 
    y_train_subset, 
    epochs=10, 
    batch_size=32, 
    verbose =2,
    validation_data=(X_test, y_test_subset)
    )

#evaluate model perfonmance

test_loss, test_accuracy = model.evaluate(X_test, y_test_subset, verbose=0)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

#visualize training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

