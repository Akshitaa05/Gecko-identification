# src/train_cnn.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import matplotlib.pyplot as plt


# ----------------- Species Names -----------------
species_names = {
    "gecko1": "Tokay Gecko (Gekko gecko)",
    "gecko2": "Leopard Gecko (Eublepharis macularius)",
    "gecko3": "Crested Gecko (Correlophus ciliatus)",
    "gecko4": "Day Gecko (Phelsuma species)",
    "gecko5": "Mediterranean House Gecko (Hemidactylus turcicus)"
}

# ----------------- Data Augmentation -----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7,1.3],
    validation_split=0.2  # reserve 20% for validation
)

train_generator = train_datagen.flow_from_directory(
    'data/augmented',  
    target_size=(224,224),
    batch_size=8,
    class_mode='categorical',
    subset='training',
)

validation_generator = train_datagen.flow_from_directory(
    'data/augmented',
    target_size=(224,224),
    batch_size=8,
    class_mode='categorical',
    subset='validation',
)

# ----------------- Transfer Learning -----------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ----------------- Callbacks -----------------
checkpoint = ModelCheckpoint('gecko_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)

# ----------------- Train -----------------
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,             # EarlyStopping will stop earlier if good enough
    callbacks=[checkpoint, earlystop]
)

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.show()


