import os
import predict
from tornado.web import RequestHandler


class MainHandler(RequestHandler):
    def get(self, *args, **kwargs):
        images = self.get_argument("imgs")
        value = self.get_argument("value")
        image_list = images.split(",")
        res = predict.test(image_list, int(value))
        data = {
            "status": 200,
            "res": res
        }
        self.write(data)

    def post(self, *args, **kwargs):
        file_img = self.request.files.get('file')
        if not os.path.exists("img_test"):
            os.mkdir("img_test")
        for img in file_img:
            with open("img_test/" + img.filename, 'wb') as f:
                f.write(img.body)

        self.write("200")
