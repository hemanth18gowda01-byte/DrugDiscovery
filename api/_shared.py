"""Shared helpers for Vercel's individual Python function routes."""

import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict

from src.server import DEFAULT_MODEL_PATH, MAX_JSON_BYTES, load_model_bundle, model_options, predict_batch


@lru_cache(maxsize=1)
def model_bundle() -> Dict[str, Any]:
    return load_model_bundle(Path(DEFAULT_MODEL_PATH))


class APIHandler(BaseHTTPRequestHandler):
    """Small response helpers shared by the serverless handlers."""

    def send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self) -> Dict[str, Any]:
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Content-Type must be application/json.")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length header is required.")
        try:
            content_length = int(raw_length)
        except ValueError as error:
            raise ValueError("Content-Length must be an integer.") from error
        if content_length < 0 or content_length > MAX_JSON_BYTES:
            raise ValueError("Request body exceeds the {} byte limit.".format(MAX_JSON_BYTES))
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Request body must contain valid UTF-8 JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("Request JSON must be an object.")
        return payload
