import os

import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

from dataset import TRAIN_DATASET_PATH, TEST_DATASET_PATH
from logs import LOGS_PATH
from models import MODELS_PATH

trained_model = tf.keras.applications.InceptionV3(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
)

for layer in trained_model.layers:
    layer.trainable = False

last_layer = trained_model.get_layer('mixed7')

last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(trained_model.input, x)

model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATASET_PATH,
    batch_size=64,
    class_mode='binary',
    target_size=(224, 224),
)

validation_generator = test_datagen.flow_from_directory(
    TEST_DATASET_PATH,
    batch_size=20,
    class_mode='binary',
    target_size=(224, 224),
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(LOGS_PATH, "inception3"))

model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=12,
    epochs=5,
    validation_steps=9,
    verbose=1,
    callbacks=[tensorboard_callback],
)

model.save(os.path.join(MODELS_PATH, "inception3.h5"))
