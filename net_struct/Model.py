# from net_struct.Inception_v3 import inceptionV3_model
from net_struct.LeNet_5 import leNet_5_model
from net_struct.AlexNet import alexNet_model
# from net_struct.ResNet import resNet_model
from net_struct.VGGNet import vggNet_model


class ModelClass:
    def __init__(self, struct, name, save_path):
        self.struct = struct
        self.name = name
        self.save_path = save_path


ModelMap = {
    "leNet_5": ModelClass(leNet_5_model, "LetNet_5", "model/LetNet_5.h5"),
    "alexNet": ModelClass(alexNet_model, "AlexNet", "model/AlexNet.h5"),
    "vggNet": ModelClass(vggNet_model, "VggNet", "model/VggNet.h5"),
    # "inceptionV3": ModelClass(inceptionV3_model, "InceptionV3", "model/InceptionV3.h5"),
    # "resNet": ModelClass(resNet_model, "ResNet", "model/ResNet.h5"),
}
