import tensorflow as tf

# 输入维度
INPUT_SHAPE = [60, 80, 3]  # 未裁剪小尺寸
# INPUT_SHAPE = [64, 64, 3]  # 裁剪的尺寸
# INPUT_SHAPE = [75, 100, 3]  # 未裁剪大尺寸
# INPUT_SHAPE = [120, 160, 3]  # 未裁剪大尺寸
# 优化器
optimizer = "adam"
# 损失
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# 记录值
metrics = ['accuracy']

epochs = 50
batch_size = 32

