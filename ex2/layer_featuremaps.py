# vgg19_brain_tumor_featuremaps_dataset.py

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model  # Graphviz installed
from matplotlib import pyplot as plt
from numpy import expand_dims
import os
import random

# ---------------------------
# 1. Load VGG19 model
# ---------------------------
model = VGG19(weights='imagenet', include_top=True)

# Plot architecture (requires Graphviz)
plot_model(model, to_file='vgg19_model_plot.png', show_shapes=True, show_layer_names=True)
print("VGG19 model loaded and architecture plotted (vgg19_model_plot.png).")

# ---------------------------
# 2. Inspect convolutional layers and filters
# ---------------------------
print("\nConvolutional layers and filter shapes:")
for layer in model.layers:
    if 'conv' not in layer.name:
        continue
    filters, biases = layer.get_weights()
    print(f"Layer: {layer.name}, Filter shape: {filters.shape}")

# ---------------------------
# 3. Visualize first few filters of the first conv layer
# ---------------------------
first_conv = model.get_layer('block1_conv1')
filters, biases = first_conv.get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters = 6
ix = 1
plt.figure(figsize=(12,6))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(3):  # RGB channels
        ax = plt.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:, :, j], cmap='gray')
        ix += 1
plt.suptitle("First 6 filters of block1_conv1 (3 channels each)")
plt.show()

# ---------------------------
# 4. Pick one random image from each class in training dataset
# ---------------------------
dataset_path = 'brain_tumor_dataset/training'  # Adjust path if needed
classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
print("\nClasses found:", classes)

sample_images = {}
for cls in classes:
    cls_folder = os.path.join(dataset_path, cls)
    img_file = random.choice(os.listdir(cls_folder))
    sample_images[cls] = os.path.join(cls_folder, img_file)

print("\nRandom images selected from each class:")
for cls, path in sample_images.items():
    print(f"{cls}: {path}")

# ---------------------------
# 5. Extract feature maps from multiple conv layers
# ---------------------------
conv_layer_names = [
    'block1_conv1', 'block1_conv2',
    'block2_conv1', 'block2_conv2',
    'block3_conv1', 'block3_conv2',
    'block4_conv1', 'block5_conv1'
]

outputs = [model.get_layer(name).output for name in conv_layer_names]
feature_model = Model(inputs=model.inputs, outputs=outputs)

# ---------------------------
# 6. Process each sample image
# ---------------------------
for cls, img_path in sample_images.items():
    print(f"\nProcessing feature maps for class '{cls}'")
    # Load and preprocess image
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    # Get feature maps
    feature_maps = feature_model.predict(img)
    
    # Visualize feature maps for each layer
    square = 8  # grid size
    for idx, fmap in enumerate(feature_maps):
        ix = 1
        plt.figure(figsize=(12,12))
        for _ in range(square):
            for _ in range(square):
                if ix > fmap.shape[-1]:
                    break
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
                ix += 1
        layer_name = conv_layer_names[idx]
        plt.suptitle(f"Feature maps of {layer_name} for {cls}")
        plt.show()
