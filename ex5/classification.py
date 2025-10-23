import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model
from io import BytesIO
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# ---------------- Load Model ---------------- #
import os
model_path = os.path.join(os.path.dirname(__file__), "DenseNet201_brain_tumor.h5")
model = load_model(model_path)

# Classes
classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
# Dataset folder names (match your testing folder structure)
test_folders = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ---------------- Evaluation ---------------- #
eval_datagen = ImageDataGenerator(rescale=1./255)
eval_dir = 'Testing/'

eval_generator = eval_datagen.flow_from_directory(
    eval_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
)

# Evaluate on full test set
loss = model.evaluate(eval_generator, steps=len(eval_generator))
for index, name in enumerate(model.metrics_names):
    print(name, loss[index])

# ---------------- Utility Functions ---------------- #
def preprocess_input(x):
    x = img_to_array(x) / 255.
    return np.expand_dims(x, axis=0)

def predict_image(im):
    x = preprocess_input(im)
    pred = np.argmax(model.predict(x))
    return pred, classes[pred]

def predict_from_image_path(image_path):
    return predict_image(load_img(image_path, target_size=(224, 224)))

def predict_from_image_url(image_url):
    res = requests.get(image_url)
    im = Image.open(BytesIO(res.content))
    return predict_image(im)

# ---------------- Grad-CAM (TF2 version) ---------------- #
def grad_CAM(image_path):
    im = load_img(image_path, target_size=(224,224))
    x = preprocess_input(im)

    # Use the last conv layer before global average pooling for DenseNet201
    # This is typically the last conv block before the final pooling
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer('conv5_block32_concat').output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        index = tf.argmax(predictions[0])
        class_channel = predictions[:, index]

    grads = tape.gradient(class_channel, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = heatmap.numpy()

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    # Overlay heatmap
    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite('./tmp.jpg', superimposed_img)

    plt.figure(figsize=(12,6))
    plt.imshow(mpimg.imread('./tmp.jpg'))
    plt.axis("off")
    plt.show()


# ---------------- Example Run ---------------- #
print(predict_from_image_path('Testing/glioma/Te-gl_0010.jpg'))
grad_CAM('Testing/glioma/Te-gl_0010.jpg')

# ---------------- Confusion Matrix + Metrics ---------------- #
print("\n=== Confusion Matrix and Metrics on Full Test Set ===")

Y_true = eval_generator.classes
Y_pred = model.predict(eval_generator, steps=len(eval_generator), verbose=1)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report (overall + per-class metrics)
report_dict = classification_report(Y_true, Y_pred_classes, target_names=classes, digits=4, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
print("\nMetrics Table (Overall + Per-Class):")
print(df_report)
df_report.to_csv("classification_metrics_full_test.csv")
print("\nSaved full test classification metrics to classification_metrics_full_test.csv")

# ---------------- Metrics for Minimum 50 Images per Class ---------------- #
indices_50_per_class = []
for i, cls in enumerate(classes):
    cls_indices = np.where(Y_true == i)[0][:50]  # first 50 images per class
    indices_50_per_class.extend(cls_indices)

Y_true_50 = Y_true[indices_50_per_class]
Y_pred_50 = Y_pred_classes[indices_50_per_class]

report_50_dict = classification_report(Y_true_50, Y_pred_50, target_names=classes, digits=4, output_dict=True)
df_report_50 = pd.DataFrame(report_50_dict).transpose()
print("\nMetrics Table (50 Images per Class):")
print(df_report_50)
df_report_50.to_csv("classification_metrics_50perclass.csv")
print("\nSaved 50-per-class metrics to classification_metrics_50perclass.csv")
