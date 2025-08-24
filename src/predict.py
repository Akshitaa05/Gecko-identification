import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load trained model
model = tf.keras.models.load_model("gecko_model.h5")

# Path to the test image(Copy the path of whichever image you want to add)
image_path = r"C:\Users\bawej\Downloads\gecko_svm_starter\aug_0_418.jpg"


# Define class labels (same order as your training folders)
class_labels = [
    "Tokay Gecko (Gekko gecko)",
    "Leopard Gecko (Eublepharis macularius)",
    "Crested Gecko (Correlophus ciliatus)",
    "Day Gecko (Phelsuma species)",
    "Mediterranean House Gecko (Hemidactylus turcicus)"
]

# Load and preprocess the image
img = load_img(image_path, target_size=(224, 224))  # must match training size
img_array = img_to_array(img) / 255.0              # normalize
img_array = np.expand_dims(img_array, axis=0)      # add batch dimension

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]
confidence = np.max(predictions) * 100

print(f"\n✅ Predicted class: {class_labels[predicted_class]} ({confidence:.2f}% confidence)")
