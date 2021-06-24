import tornado.web
import config
import index


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/index', index.MainHandler),
        ]
        super(Application, self).__init__(handlers, **config.settings)
