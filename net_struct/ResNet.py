import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from model_config import INPUT_SHAPE
resNet_base = tf.keras.applications.ResNet50(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet')
resNet_base.trainable = True
# for layer in resNet_base.layers[:-100]:
#     layer.trainable = False
resNet_model = models.Sequential()
resNet_model.add(resNet_base)
resNet_model.add(layers.Flatten())
resNet_model.add(layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.1)))
layers.Dropout(0.6),
resNet_model.add(layers.Dense(2, activation=None))
