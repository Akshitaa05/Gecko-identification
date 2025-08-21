from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7,1.3]
)

# Paths
original_data_dir = 'data/'       # Original images
augmented_data_dir = 'data/augmented/'  # Folder to save augmented images

# Make sure augmented folders exist
species = ['gecko1','gecko2','gecko3','gecko4','gecko5']
for s in species:
    os.makedirs(os.path.join(augmented_data_dir, s), exist_ok=True)

# Number of augmented images per original image
aug_per_image = 5

# Generate and save
for s in species:
    folder = os.path.join(original_data_dir, s)
    save_folder = os.path.join(augmented_data_dir, s)
    
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        
        # Load image
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        img = load_img(img_path, target_size=(224,224))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # reshape to (1, 224,224,3)
        
        # Create augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_folder,
                                  save_prefix='aug', save_format='jpg'):
            i += 1
            if i >= aug_per_image:
                break  # Stop after generating aug_per_image images per original
