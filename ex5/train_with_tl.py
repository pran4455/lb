import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet201
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import optimizers
import matplotlib.pyplot as plt

# Paths to dataset
train_dir = 'Training/'
test_dir  = 'Testing/'

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Create base model (DenseNet201)
base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Add new trainable layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

adam = optimizers.Adam(lr=0.0001)

# Compile model
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit_generator(train_generator, verbose=1,
                              steps_per_epoch=len(train_generator),
                              epochs=2,
                              validation_data=test_generator,
                              validation_steps=len(test_generator))

# Save model
model.save("DenseNet201_brain_tumor.h5")

# Plot training history
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'r-', label='Validation acc')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.legend()
    plt.show()
    return acc, val_acc, loss, val_loss

acc, val_acc, loss, val_loss = plot_history(history)
