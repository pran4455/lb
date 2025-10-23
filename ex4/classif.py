# ex4_brain_tumor_classification_v2.py

# import necessary packages
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import argparse
import cv2

# construct the argument parse and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="path to input image for prediction")
ap.add_argument("-t", "--train", action="store_true", help="set this flag to train the model")
args = vars(ap.parse_args())

# define dataset paths
train_dir = "brain_tumor_dataset/training"
test_dir = "brain_tumor_dataset/testing"

# image parameters
img_width, img_height = 224, 224
batch_size = 32
num_classes = len(os.listdir(train_dir))  # automatically get number of classes
model_path = "vgg19_brain_tumor.h5"

# function to build the model
def build_model():
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ----------------------
# Training the model
# ----------------------
if args["train"]:
    print("[INFO] Preparing data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print("[INFO] Building model...")
    model = build_model()

    print("[INFO] Training the model...")
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=test_generator,
        validation_steps=test_generator.samples // batch_size,
        epochs=5
    )

    model.save(model_path)
    print(f"[INFO] Model trained and saved as {model_path}")

# ----------------------
# Predicting a single image
# ----------------------
if args["image"]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. Train first using --train flag.")

    print("[INFO] Loading trained model...")
    model = load_model(model_path)

    print("[INFO] Loading and preprocessing image...")
    image = load_img(args["image"], target_size=(img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    print("[INFO] Classifying image...")
    preds = model.predict(image)
    class_idx = np.argmax(preds, axis=1)[0]
    class_labels = list(os.listdir(train_dir))
    label = class_labels[class_idx]
    confidence = preds[0][class_idx] * 100

    # Reduce confidence for display (max 0 to avoid negative)
    display_confidence = max(confidence - 10, 0)

    print(f"[RESULT] Predicted Class: {label} ({display_confidence:.2f}%)")

    # display image with modified confidence
    orig = cv2.imread(args["image"])
    cv2.putText(orig, f"Label: {label}, {display_confidence:.2f}%",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Classification", orig)
    cv2.waitKey(0)