import os

from net_struct.Model import ModelMap

BASE_DIRS = os.path.dirname(__file__)

# 数据集文件位置
DATASET_PATH = BASE_DIRS + r"\dataset\Data\\"

# 类别
CLASSIFY = ["false", "true"]

# 类别和类别编号对应表
CLASS_NUMBER_FILE = BASE_DIRS + r"\dataset\file\classify_number.txt"
# 图片和类别编号对应表
PIC_NUMBEI_FILE = BASE_DIRS + r"\dataset\file\picture_number.txt"

# 训练集占比
RATE = 0.8

# 选择模型
# model = ModelMap["leNet_5"]
model = ModelMap["alexNet"]
# model = ModelMap["vggNet"]
# model = ModelMap["inceptionV3"]
# model = ModelMap["resNet"]

# 参数
options = {
    'port': 8080,
}

# 配置
settings = {
    "static_path": os.path.join(BASE_DIRS, 'static'),
    "template_path": os.path.join(BASE_DIRS, 'templates'),
    "cookie_secret": '2LBKQd6iTOWKBlPiyvXG+1aTW0PdDEHsmkb4s+Nzfcs=',
    # 安全cookie的秘钥,uuid base64加密获得  base64.b64encode(uuid.uuid4().bytes + uuid.uuid4.bytes)
    "xsrf_cookies": False,  # True为开启同源保护
    # "login_url": '/login',  # authenticated指定的重定向页面
    "debug": True,
    # "autoescape": None 该设置为关闭整个项目的自动转义,不推荐使用
}
