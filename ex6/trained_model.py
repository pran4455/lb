import h5py
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# -----------------------------
# Load features
# -----------------------------
def load_features(path):
    h5f = h5py.File(path, 'r')
    batch_count = h5f['batches'][()]
    X_list, y_list = [], []
    for batch_id in range(batch_count):
        X_list.append(np.array(h5f[f'features-{batch_id}']))
        y_list.append(np.array(h5f[f'labels-{batch_id}']))
    X_all = np.vstack(X_list)
    y_all = np.vstack(y_list)
    h5f.close()
    return X_all, y_all

X_train, y_train = load_features('./data/train_brain.h5')
X_val, y_val = load_features('./data/validation_brain.h5')

# -----------------------------
# Model
# -----------------------------
inputs = Input(shape=(7,7,1920))  # DenseNet201 output dimensions
x = GlobalAveragePooling2D()(inputs)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(4, activation='softmax')(x)
model = Model(inputs, outputs)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# Optional: class weights
# -----------------------------
y_train_labels = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights = dict(enumerate(class_weights))

# -----------------------------
# Train
# -----------------------------
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=32,
                    epochs=50,
                    class_weight=class_weights)

# -----------------------------
# Save model
# -----------------------------
model.save('./output/densenet201_brain_model.h5')

# -----------------------------
# Plot accuracy & loss
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
