"""Serve the prediction UI and JSON API for the trained NLM model."""

import json
import logging
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import joblib
from scipy.sparse import csr_matrix

from train_surveillance import parse_genotypes

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
DEFAULT_MODEL_PATH = ROOT / "models" / "nlm_resistance_model.joblib"
MAX_JSON_BYTES = 5 * 1024 * 1024
MAX_BATCH_ROWS = 250
LOGGER = logging.getLogger("card_api")


class CardHTTPServer(ThreadingHTTPServer):
    model_bundle: Dict[str, Any]


def load_model_bundle(model_path: Path) -> Dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError("Model bundle does not exist: {}".format(model_path))
    bundle = joblib.load(model_path)
    required = {"vectorizer", "classifier", "target_classes"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(
            "Model bundle is missing required entries: {}".format(
                ", ".join(sorted(missing))
            )
        )
    return bundle


def model_options(bundle: Dict[str, Any]) -> Dict[str, List[str]]:
    feature_names = bundle["vectorizer"].feature_names_
    organisms = sorted(
        name[len("organism::") :]
        for name in feature_names
        if name.startswith("organism::")
    )
    antibiotics = sorted(
        name[len("antibiotic::") :]
        for name in feature_names
        if name.startswith("antibiotic::")
    )
    return {"organisms": organisms, "antibiotics": antibiotics}


def predict_batch(
    bundle: Dict[str, Any], payload: Dict[str, Any]
) -> Dict[str, Any]:
    antibiotic = payload.get("antibiotic")
    samples = payload.get("samples")
    if not isinstance(antibiotic, str) or not antibiotic.strip():
        raise ValueError("Provide one antibiotic name for this prediction batch.")
    antibiotic = antibiotic.strip()
    if not isinstance(samples, list) or not samples:
        raise ValueError("Provide at least one isolate row.")
    if len(samples) > MAX_BATCH_ROWS:
        raise ValueError(
            "A request may contain at most {} isolates.".format(MAX_BATCH_ROWS)
        )

    options = model_options(bundle)
    known_organisms = set(options["organisms"])
    known_antibiotics = set(options["antibiotics"])
    vectorizer = bundle["vectorizer"]
    known_features = set(vectorizer.feature_names_)
    row_features: List[Dict[str, float]] = []
    row_warnings: List[List[str]] = []
    identifiers: List[str] = []

    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError("Isolate row {} must be a JSON object.".format(index))
        organism = sample.get("organism")
        genotype_text = sample.get("genotypes", "")
        identifier = sample.get("id", "row-{}".format(index))
        if not isinstance(organism, str) or not organism.strip():
            raise ValueError(
                "Isolate row {} has no organism value.".format(index)
            )
        if not isinstance(genotype_text, str):
            raise ValueError(
                "Genotype calls for row {} must be text.".format(index)
            )
        if not isinstance(identifier, str):
            identifier = str(identifier)

        organism = organism.strip()
        genotype_names = parse_genotypes(genotype_text, identifier)
        features = {feature_name: 1.0 for feature_name in genotype_names}
        organism_feature = "organism::{}".format(organism)
        antibiotic_feature = "antibiotic::{}".format(antibiotic)
        features[organism_feature] = 1.0
        features[antibiotic_feature] = 1.0

        warnings: List[str] = []
        if not genotype_names:
            warnings.append(
                "No AMR genotype calls were supplied; prediction uses organism "
                "and antibiotic only."
            )
        elif not any(
            name in known_features
            for name in genotype_names
            if name.startswith("gene::")
        ):
            warnings.append(
                "None of the supplied AMR genes were learned by this model."
            )
        unknown_states = [
            name
            for name in genotype_names
            if name.startswith("call::") and name not in known_features
        ]
        if unknown_states:
            warnings.append(
                "{} genotype call state(s) were not learned by the model.".format(
                    len(unknown_states)
                )
            )
        if organism not in known_organisms:
            warnings.append(
                "Organism is not represented in model training: {}.".format(
                    organism
                )
            )
        if antibiotic not in known_antibiotics:
            warnings.append(
                "Antibiotic is not represented in model training: {}.".format(
                    antibiotic
                )
            )

        row_features.append(features)
        row_warnings.append(warnings)
        identifiers.append(identifier)

    matrix = csr_matrix(vectorizer.transform(iter(row_features)))
    classifier = bundle["classifier"]
    predictions = classifier.predict(matrix)
    probabilities = classifier.predict_proba(matrix)
    class_names = [str(class_name) for class_name in classifier.classes_]

    results = []
    for index, (prediction, probability) in enumerate(
        zip(predictions, probabilities)
    ):
        prediction = str(prediction)
        probability_by_class = {
            class_name: float(score)
            for class_name, score in zip(class_names, probability)
        }
        results.append(
            {
                "id": identifiers[index],
                "prediction": prediction,
                "prediction_label": (
                    "resistant" if prediction == "R" else "susceptible"
                ),
                "probabilities": probability_by_class,
                "organism": samples[index]["organism"].strip(),
                "antibiotic": antibiotic,
                "genotype_calls_supplied": len(
                    [
                        name
                        for name in row_features[index]
                        if name.startswith("gene::")
                    ]
                ),
                "note": bundle.get(
                    "note",
                    "Research prototype; not validated for clinical treatment decisions.",
                ),
                "warnings": row_warnings[index],
            }
        )

    resistant = sum(result["prediction"] == "R" for result in results)
    return {
        "antibiotic": antibiotic,
        "count": len(results),
        "summary": {
            "resistant": resistant,
            "susceptible": len(results) - resistant,
        },
        "results": results,
        "note": bundle.get(
            "note",
            "Research prototype; not validated for clinical treatment decisions.",
        ),
    }


class CardRequestHandler(SimpleHTTPRequestHandler):
    server_version = "CARDGenomeApp/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    @property
    def model_bundle(self) -> Dict[str, Any]:
        return self.server.model_bundle  # type: ignore[attr-defined]

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
            "style-src 'self'; img-src 'self' data:; connect-src 'self'; "
            "worker-src 'self' blob:; "
            "object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
        )
        super().end_headers()

    def send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self) -> Dict[str, Any]:
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json.")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length header is required.")
        try:
            content_length = int(raw_length)
        except ValueError as error:
            raise ValueError("Content-Length must be an integer.") from error
        if content_length < 0 or content_length > MAX_JSON_BYTES:
            raise ValueError(
                "Request body exceeds the {} byte limit.".format(MAX_JSON_BYTES)
            )
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Request body must contain valid UTF-8 JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("Request JSON must be an object.")
        return payload

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/health":
            self.send_json(
                200,
                {
                    "status": "ok",
                    "api": "online",
                    "model": "loaded",
                    "model_version": self.model_bundle.get(
                        "scikit_learn_version", "unknown"
                    ),
                    "feature_count": len(
                        self.model_bundle["vectorizer"].feature_names_
                    ),
                },
            )
            return
        if route == "/api/options":
            self.send_json(200, model_options(self.model_bundle))
            return
        if route.startswith("/api/"):
            self.send_json(404, {"error": "API endpoint not found."})
            return
        super().do_GET()

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route != "/api/predict":
            self.send_json(404, {"error": "API endpoint not found."})
            return
        try:
            payload = self.read_json_body()
            response = predict_batch(self.model_bundle, payload)
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
            return
        except BrokenPipeError:
            LOGGER.info("Prediction request was closed by the client.")
            return
        self.send_json(200, response)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    model_path = Path(
        os.environ.get("CARD_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    ).resolve()
    model_bundle = load_model_bundle(model_path)
    host = os.environ.get("CARD_HOST", "127.0.0.1")
    port = int(os.environ.get("CARD_PORT", "8000"))
    server = CardHTTPServer((host, port), CardRequestHandler)
    server.model_bundle = model_bundle
    LOGGER.info(
        "Serving UI and API at http://%s:%s using %s",
        host,
        port,
        model_path,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
