import os
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from matplotlib import pyplot as plt

# Path to training images
train_path = "brain_tumor_dataset/training"

# Collect first 5 image file paths across all subfolders
image_files = []
for class_folder in os.listdir(train_path):
    class_path = os.path.join(train_path, class_folder)
    if os.path.isdir(class_path):
        files = [os.path.join(class_path, f) 
                 for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg','.png'))]
        image_files.extend(files)
image_files = image_files[:5]  # pick first 5 images for demo

# Function to apply augmentation and show images
def show_augmented_images(img_path, datagen, title="", rows=3, cols=3):
    img = load_img(img_path, target_size=(224,224))  # Resize for display
    data = img_to_array(img)
    samples = expand_dims(data, 0)
    it = datagen.flow(samples, batch_size=1)
    
    plt.figure(figsize=(6,6))
    plt.suptitle(title)
    for i in range(rows * cols):
        batch = next(it)  # updated for new Keras versions
        image = batch[0].astype('uint8')
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.show()

# 1. Width Shift
datagen_width = ImageDataGenerator(width_shift_range=[-50,50])
for img_path in image_files:
    show_augmented_images(img_path, datagen_width, title="Width Shift Augmentation")

# 2. Height Shift
datagen_height = ImageDataGenerator(height_shift_range=0.2)
for img_path in image_files:
    show_augmented_images(img_path, datagen_height, title="Height Shift Augmentation")

# 3. Horizontal Flip
datagen_flip = ImageDataGenerator(horizontal_flip=True)
for img_path in image_files:
    show_augmented_images(img_path, datagen_flip, title="Horizontal Flip Augmentation")

# 4. Rotation
datagen_rotate = ImageDataGenerator(rotation_range=45)
for img_path in image_files:
    show_augmented_images(img_path, datagen_rotate, title="Rotation Augmentation")

# 5. Brightness Adjustment
datagen_brightness = ImageDataGenerator(brightness_range=[0.5,1.5])
for img_path in image_files:
    show_augmented_images(img_path, datagen_brightness, title="Brightness Augmentation")

# 6. Zoom
datagen_zoom = ImageDataGenerator(zoom_range=[0.7,1.2])
for img_path in image_files:
    show_augmented_images(img_path, datagen_zoom, title="Zoom Augmentation")
