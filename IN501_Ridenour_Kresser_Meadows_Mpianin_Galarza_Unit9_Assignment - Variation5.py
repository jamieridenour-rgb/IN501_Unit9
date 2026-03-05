"""
Program Name: IN501_Ridenour_Kresser_Meadows_Mpianin_Galarza_Unit9_Assignment - Variation5.py
Author: Jamie Ridenour, Robert Kresser, Lane Meadows, Francisca Mpianin, Marie Galarza
Date: 03-05-2026
Description:
# Unit 9 – Group Project, MNIST dataset
Image comes up, you close it, it will start to perform the Epoch's then at the very end the test predictions will pop up.
Add Extra Convolution Layer
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape((60000,28,28,1))
test_images = test_images.reshape((10000,28,28,1))

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128,(3,3),activation='relu'),  # new layer
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=5)

test_loss,test_acc=model.evaluate(test_images,test_labels)
print("Test Loss:",test_loss)
print("Test Accuracy:",test_acc)

predictions=model.predict(test_images)

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(test_images[i].reshape(28,28),cmap='gray')
    plt.title("Pred:"+str(np.argmax(predictions[i])))
    plt.axis('off')

plt.show()