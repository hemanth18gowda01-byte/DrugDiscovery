from api._shared import APIHandler, model_bundle
from src.server import predict_batch


class handler(APIHandler):
    def do_POST(self):
        try:
            self.send_json(200, predict_batch(model_bundle(), self.read_json_body()))
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
