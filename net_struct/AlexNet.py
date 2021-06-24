from tensorflow.keras import models, layers

import model_config

INPUT_SHAPE = model_config.INPUT_SHAPE

alexNet_model = models.Sequential([
    layers.Conv2D(filters=48, kernel_size=11, strides=4, padding="same", activation="relu", input_shape=INPUT_SHAPE),
    layers.BatchNormalization(name='BN1'),
    layers.MaxPool2D(pool_size=3, strides=2),
    layers.Conv2D(filters=128, kernel_size=5, padding='same', activation='relu'),
    layers.BatchNormalization(name='BN2'),
    layers.MaxPool2D(pool_size=3, strides=2),
    layers.Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'),
    layers.Conv2D(filters=86, kernel_size=3, padding='same', activation='relu'),
    layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    layers.BatchNormalization(name='BN3'),
    layers.MaxPool2D(pool_size=3, strides=2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='sigmoid')
])
