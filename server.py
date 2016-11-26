from StringIO import StringIO
import json
import logging
import os

import numpy
from PIL import Image
import tornado.escape
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket

import image_functions

tornado.options.define("port",
                       default=8888,
                       help="run on the given port",
                       type=int)


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [(r"/", SetupHarvestHandler),
                    (r"/harvesting", HarvestHandler),
                    (r"/predict", PredictHandler),
                    (r"/train", TrainHandler),
                    ]
        file_path = os.path.dirname(__file__)
        settings = dict(
            cookie_secret="asdsafl.rleknknfkjqweonrkbknoijsdfckjnk 234jn",
            template_path=os.path.join(file_path, "templates"),
            static_path=os.path.join(file_path, "static"),
            xsrf_cookies=False,
            autoescape=None,
            debug=True
        )
        tornado.web.Application.__init__(self, handlers, **settings)


class SocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        logging.info('new connection')

    def on_message(self, message):
        message_file_object = StringIO(message)
        image = Image.open(message_file_object)
        cv_image = numpy.array(image)
        self.process(cv_image)

    def on_close(self):
        logging.info('connection closed')

    def process(self, cv_image):
        pass


class SetupHarvestHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("harvest.html")

    def post(self):
        name = self.get_argument("label", None)
        if not name:
            logging.error("No label, bailing out")
            return
        logging.info("Got label %s" % name)
        image_functions.Label.get_or_create(name=name)[0].persist()
        logging.info("Setting secure cookie %s" % name)
        self.set_secure_cookie('label', name)
        self.redirect("/")


class HarvestHandler(SocketHandler):
    def process(self, cv_image):
        label = image_functions.Label.get(
            image_functions.Label.name == self.get_secure_cookie('label'))
        logging.info("Got label: %s" % label.name)
        if not label:
            logging.info("No cookie, bailing out")
            return
        logging.info("About to save image")
        result = image_functions.Image(label=label).persist(cv_image)
        if result == 'Done':
            self.write_message(json.dumps(result))


class PredictHandler(SocketHandler):
    def process(self, cv_image):
        result = image_functions.predict(cv_image)
        if result:
            self.write_message(json.dumps(result))


class TrainHandler(tornado.web.RequestHandler):
    def post(self):
        image_functions.train()


def main():
    tornado.options.parse_command_line()
    image_functions.Image().delete()
    logging.info("Images deleted")
    image_functions.Label().delete()
    logging.info("Labels deleted")
    image_functions.load_images_to_db("data/images")
    logging.info("Labels and images loaded")
    image_functions.train()
    logging.info("Model trained")
    app = Application()
    app.listen(tornado.options.options.port)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
