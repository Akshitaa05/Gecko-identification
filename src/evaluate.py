import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Load trained model
model = tf.keras.models.load_model("gecko_model.h5")

# Path to your original dataset (organised in subfolders by class)
test_dir = r"C:\Users\bawej\Downloads\gecko_svm_starter\data\original"

# Create test generator (must match training size, e.g. 224x224)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),   # keep SAME size as training
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate model on original dataset
loss, acc = model.evaluate(test_generator)
print(f"\n✅ Accuracy on original dataset: {acc*100:.2f}%")
