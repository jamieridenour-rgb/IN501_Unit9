"""
Program Name: IN501_Ridenour_Kresser_Meadows_Mpianin_Galarza_Unit9_Assignment - Variation2.py
Author: Jamie Ridenour, Robert Kresser, Lane Meadows, Francisca Mpianin, Marie Galarza
Date: 03-05-2026
Description:
# Unit 9 – Group Project, MNIST dataset
Image comes up, you close it, it will start to perform the Epoch's then at the very end the test predictions will pop up.
Add Dropout
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

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape for CNN input
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Display first 10 images
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title("Label: " + str(train_labels[i]))
    plt.axis('off')
plt.show()

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Predictions
predictions = model.predict(test_images)

# Display predictions for first 10 test images
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(test_images[i].reshape(28,28), cmap='gray')
    predicted_label = np.argmax(predictions[i])
    plt.title("Pred: " + str(predicted_label))
    plt.axis('off')
plt.show()