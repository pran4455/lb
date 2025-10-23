import os
import h5py
import numpy as np
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

# Load pre-trained DenseNet201 (exclude top)
conv_base = DenseNet201(weights='imagenet', include_top=False, input_shape=(224,224,3))

def extract_features(file_name, directory, batch_size=32, target_size=(224,224)):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    h5_file = h5py.File(file_name, 'w')
    
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(directory,
                                            target_size=target_size,
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            shuffle=False)  # important: no shuffle
    
    total_samples = generator.samples
    steps = math.ceil(total_samples / batch_size)
    
    print(f"Total samples: {total_samples}, batch size: {batch_size}, steps: {steps}")
    
    samples_processed = 0
    for batch_number in range(steps):
        inputs_batch, labels_batch = next(generator)
        features_batch = conv_base.predict(inputs_batch)
        
        # Save batch features and labels
        h5_file.create_dataset(f'features-{batch_number}', data=features_batch)
        h5_file.create_dataset(f'labels-{batch_number}', data=labels_batch)
        
        samples_processed += inputs_batch.shape[0]
        print(f"Processed batch {batch_number+1}/{steps} | Total samples processed: {samples_processed}", end='\r')
    
    # Save number of batches
    h5_file.create_dataset('batches', data=steps)
    h5_file.close()
    print(f"\nFeature extraction completed for {directory}")

# -----------------------------
# Run for train, validation, and test
# -----------------------------
extract_features('./data/train_brain.h5', './BrainTumorDataset/train', batch_size=32)
extract_features('./data/validation_brain.h5', './BrainTumorDataset/validation', batch_size=32)
extract_features('./data/test_brain.h5', './BrainTumorDataset/test', batch_size=32)
