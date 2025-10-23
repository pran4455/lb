import os
import shutil
import random

source_dir = "Training"
target_base = "./BrainTumorDataset"
validation_split = 0.15
classes = ["glioma", "meningioma", "pituitary", "notumor"]

for cls in classes:
    class_dir = os.path.join(source_dir, cls)
    images = os.listdir(class_dir)
    random.shuffle(images)
    
    n_val = int(len(images) * validation_split)
    
    os.makedirs(os.path.join(target_base, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(target_base, "validation", cls), exist_ok=True)
    os.makedirs(os.path.join(target_base, "test", cls), exist_ok=True)
    
    for i, img in enumerate(images):
        src = os.path.join(class_dir, img)
        if i < n_val:
            dst = os.path.join(target_base, "validation", cls, img)
        else:
            dst = os.path.join(target_base, "train", cls, img)
        shutil.copyfile(src, dst)

# Copy Testing folder
for cls in classes:
    test_class_dir = os.path.join("Testing", cls)
    for img in os.listdir(test_class_dir):
        src = os.path.join(test_class_dir, img)
        dst = os.path.join(target_base, "test", cls, img)
        shutil.copyfile(src, dst)
