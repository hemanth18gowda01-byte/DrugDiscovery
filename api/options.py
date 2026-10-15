from api._shared import APIHandler, model_bundle
from src.server import model_options


class handler(APIHandler):
    def do_GET(self):
        self.send_json(200, model_options(model_bundle()))
