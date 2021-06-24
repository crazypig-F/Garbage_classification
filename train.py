import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
import model_config
from dataset import garbage
from dataset import get_image_label_list

INPUT_SHAPE = model_config.INPUT_SHAPE
optimizer = model_config.optimizer
loss = model_config.loss
metrics = model_config.metrics
epoch = model_config.epochs
batch_size = model_config.batch_size

MODEL_SAVE_PATH = config.model.save_path
MODEL_NAME = config.model.name
model = config.model.struct

# 加载训练集和测试集
images, labels = get_image_label_list.get()
train_ds = garbage.load_garbage("train", images, labels)
train_ds = iter(train_ds)
train_images, train_labels = next(train_ds)

test_ds = garbage.load_garbage("test", images, labels)
test_ds = iter(test_ds)
test_images, test_labels = next(test_ds)

# 如果模型不存在就训练模型
if not os.path.exists("model"):
    os.mkdir("model")
if not os.path.exists(MODEL_SAVE_PATH):
    model.summary()
    # 配置优化方法
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    log_dir = os.path.join('logs', MODEL_NAME)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # 训练模型
    history = model.fit(train_images, train_labels, epochs=epoch,
                        validation_data=(test_images, test_labels),
                        batch_size=batch_size,
                        callbacks=[tensorboard])
    # 保存模型
    model.save(MODEL_SAVE_PATH)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

else:
    # 加载保存的模型
    model = keras.models.load_model(MODEL_SAVE_PATH)
    # 从测试集中随机抽取10张图片，打印测试结果
    for i in range(10):
        k = np.random.randint(low=0, high=len(test_images))
        pred = model.predict(tf.convert_to_tensor([test_images[k]]))
        print("预测结果为", tf.argmax(pred[0]).numpy(), "，标签是", test_labels[k].numpy())



