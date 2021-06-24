import os

from tensorflow import keras
import tensorflow as tf
import config
import model_config
MODEL_DIR = config.model.save_path
MODEL_NAME = config.model.name
INPUT_SHAPE = model_config.INPUT_SHAPE


def test(img_list, model_type):
    # 加载保存的模型
    if model_type == 1:
        model_dir = "./model/LetNet_5.h5"
    elif model_type == 2:
        model_dir = "./model/AlexNet.h5"
    elif model_type == 3:
        model_dir = "./model/VggNet.h5"
    elif model_type == 4:
        model_dir = "./model/LetNet_5_crop.h5"
    elif model_type == 5:
        model_dir = "./model/AlexNet_crop.h5"
    else:
        model_dir = "./model/VggNet_crop.h5"
    model = keras.models.load_model(model_dir)
    images = preprocess(img_list, model_type)
    res = []
    for img in images:
        pred = model.predict(tf.convert_to_tensor([img]))
        res.append(str(tf.argmax(pred[0]).numpy()))
    return res


def preprocess(img_list, model_type):
    images = []
    for name in img_list:
        x = tf.io.read_file("./img_test/" + name)  # 根据路径读取图片
        x = tf.image.decode_jpeg(x, channels=3)  # 图片解码
        if model_type == 1 or model_type == 2 or model_type == 3:
            x = tf.image.resize(x, [60, 80])  # 图片缩放
        else:
            x = tf.image.resize(x, [64, 64])

        # 数据增强
        # x = tf.image.random_flip_up_down(x)
        # x= tf.image.random_flip_left_right(x) # 左右镜像
        # x = tf.image.random_crop(x, INPUT_SHAPE) # 随机裁剪

        # 转换成张量
        # x: [0,255]=> 0~1 归一化
        x = tf.cast(x, dtype=tf.float32) / 255.
        images.append(x)
    return images


if __name__ == '__main__':
    img = os.listdir("./img/false_crop/")
    a = test(img, 6)
    n = 0
    for index, i in enumerate(a):
        if i == '0':
            n += 1
        else:
            print(index+1, img[index])
    print(n, len(a))
    print(n/len(a))
