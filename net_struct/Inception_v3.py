import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from model_config import INPUT_SHAPE
googleNet_base = tf.keras.applications.InceptionV3(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet')
googleNet_base.trainable = True
# for layer in googleNet_base.layers[:-100]:
#     layer.trainable = False
inceptionV3_model = models.Sequential()
inceptionV3_model.add(googleNet_base)
inceptionV3_model.add(layers.Flatten())
inceptionV3_model.add(layers.Dense(256, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.1)))
layers.Dropout(0.5),
inceptionV3_model.add(layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.1)))
layers.Dropout(0.5),
inceptionV3_model.add(layers.Dense(2, activation=None))
