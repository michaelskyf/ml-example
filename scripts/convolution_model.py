import os

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sys
os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))

from dataset import TRAIN_DATASET_PATH, TEST_DATASET_PATH
from logs import LOGS_PATH
from models import MODELS_PATH

model = Sequential([
    Conv2D(100, (3, 3), activation='relu', input_shape=(200, 200, 1)),
    MaxPooling2D(2, 2),

    Conv2D(100, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATASET_PATH,
    batch_size=256,
    target_size=(200, 200),
    color_mode="grayscale",
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = validation_datagen.flow_from_directory(
    TEST_DATASET_PATH,
    batch_size=256,
    target_size=(200, 200),
    color_mode="grayscale",
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(LOGS_PATH, "convolution"))

history = model.fit_generator(
    train_generator,
    epochs=5,
    validation_data=validation_generator,
    callbacks=[tensorboard_callback]
)

model.save(os.path.join(MODELS_PATH, "convolution_model_2.h5"))
