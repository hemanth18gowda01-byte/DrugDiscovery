from api._shared import APIHandler, model_bundle


class handler(APIHandler):
    def do_GET(self):
        bundle = model_bundle()
        self.send_json(200, {
            "status": "ok", "api": "online", "model": "loaded",
            "model_version": bundle.get("scikit_learn_version", "unknown"),
            "feature_count": len(bundle["vectorizer"].feature_names_),
        })
