import h5py
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load trained model
# ---------------------------
model_path = './output/densenet201_brain_model.h5'
model = load_model(model_path)
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

# -----------------------------
# Load test features
# -----------------------------
def load_test_features(path):
    h5f = h5py.File(path, 'r')
    batch_count = h5f['batches'][()]
    X_list, y_list = [], []
    for batch_id in range(batch_count):
        X_list.append(np.array(h5f[f'features-{batch_id}']))
        y_list.append(np.array(h5f[f'labels-{batch_id}']))
    X_all = np.vstack(X_list)
    y_all = np.hstack([np.argmax(y, axis=1) for y in y_list])
    h5f.close()
    return X_all, y_all

X_test, y_test = load_test_features('./data/test_brain.h5')
print(f"Loaded {X_test.shape[0]} test samples.")

# -----------------------------
# Per-class metrics (≥50 images per class)
# -----------------------------
per_class_metrics = []
for i, class_name in enumerate(classes):
    idx = np.where(y_test == i)[0]
    if len(idx) < 50:
        print(f"Warning: class '{class_name}' has less than 50 images ({len(idx)}). Using all available samples.")
    idx = idx[:50]
    X_class = X_test[idx]
    y_true_class = y_test[idx]
    
    y_pred_class = np.argmax(model.predict(X_class), axis=1)
    
    acc = accuracy_score(y_true_class, y_pred_class)
    prec = precision_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    rec = recall_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    f1 = f1_score(y_true_class, y_pred_class, average='macro', zero_division=0)
    
    per_class_metrics.append([class_name, acc, prec, rec, f1])

per_class_df = pd.DataFrame(per_class_metrics, columns=['Class','Accuracy','Precision','Recall','F1-score'])
print("\nPer-class metrics (≥50 images per class):")
print(per_class_df)

# -----------------------------
# Overall metrics
# -----------------------------
y_pred_all = np.argmax(model.predict(X_test), axis=1)
overall_acc = accuracy_score(y_test, y_pred_all)
overall_prec = precision_score(y_test, y_pred_all, average='macro', zero_division=0)
overall_rec = recall_score(y_test, y_pred_all, average='macro', zero_division=0)
overall_f1 = f1_score(y_test, y_pred_all, average='macro', zero_division=0)

overall_df = pd.DataFrame([['Overall', overall_acc, overall_prec, overall_rec, overall_f1]],
                          columns=['Class','Accuracy','Precision','Recall','F1-score'])
print("\nOverall metrics (all test samples):")
print(overall_df)

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred_all)
print("\nConfusion Matrix (overall):")
print(cm)

plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Overall)")
plt.colorbar()
plt.xticks(np.arange(len(classes)), classes, rotation=45)
plt.yticks(np.arange(len(classes)), classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# -----------------------------
# Save metrics CSV
# -----------------------------
os.makedirs('./output', exist_ok=True)
per_class_df.to_csv('./output/per_class_metrics.csv', index=False)
overall_df.to_csv('./output/overall_metrics.csv', index=False)
print("\nMetrics saved to ./output/per_class_metrics.csv and ./output/overall_metrics.csv")
