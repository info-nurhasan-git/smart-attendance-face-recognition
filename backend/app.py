from __future__ import annotations

import base64
import binascii
import os
import pickle
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from flask import Flask, Response, jsonify, request
from PIL import Image, UnidentifiedImageError
from sklearn.exceptions import InconsistentVersionWarning

import database as db
from embeddings import EMBEDDING_DIM, extract_embedding

# The pickle was created with scikit-learn 1.6.1; newer 1.x versions load it
# safely. Suppress only that specific notice to keep the API logs clean.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model" / "attendance_model.pkl"
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def load_model() -> Tuple[Dict[str, Any] | None, str | None]:
    if not MODEL_PATH.exists():
        return None, f"Model file not found at {MODEL_PATH}"
    try:
        with MODEL_PATH.open("rb") as handle:
            bundle = pickle.load(handle)
        classifier = bundle["classifier"]
        encoder = bundle["label_encoder"]
        if int(classifier.n_features_in_) != EMBEDDING_DIM:
            return None, (
                f"Model expects {classifier.n_features_in_} features, "
                f"extractor produces {EMBEDDING_DIM}."
            )
        return {"classifier": classifier, "encoder": encoder}, None
    except Exception as exc:  # pragma: no cover - defensive startup guard
        return None, f"Could not load model: {exc}"


MODEL, MODEL_ERROR = load_model()


def class_name(class_label: Any) -> str:
    """Map a classifier output to a name (the pickle encodes names as 0/1)."""
    if MODEL is None:
        return str(class_label)
    classes = list(MODEL["encoder"].classes_)
    if class_label in classes:
        return str(class_label)
    try:
        index = int(class_label)
    except (TypeError, ValueError):
        return str(class_label)
    return str(classes[index]) if 0 <= index < len(classes) else str(class_label)


def pkl_prediction(vector: np.ndarray) -> Dict[str, Any] | None:
    """Secondary signal from the supplied 2-person pickle; never blocks a scan."""
    if MODEL is None:
        return None
    try:
        classifier = MODEL["classifier"]
        labels = list(classifier.classes_)
        probabilities = classifier.predict_proba(vector.reshape(1, -1))[0]
        best = int(np.argmax(probabilities))
        return {
            "guess": class_name(labels[best]),
            "confidence": round(float(probabilities[best]), 4),
        }
    except Exception:
        return None


def request_data() -> Tuple[Dict[str, Any], Any]:
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        return payload, None
    return {}, request.files


def decode_image(payload: Dict[str, Any], files: Any) -> tuple[Image.Image, bytes, str]:
    """Return (PIL image, raw bytes, content type) from JSON or multipart."""
    raw: bytes | None = None
    content_type = "image/jpeg"

    if payload.get("image"):
        value = str(payload["image"])
        if value.lstrip().startswith("data:"):
            header, value = value.split(",", 1)
            content_type = header[5:].split(";")[0] or "image/jpeg"
        try:
            raw = base64.b64decode(value, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64 image data.") from exc
    elif files:
        uploaded = files.get("file") or next(iter(files.values()))
        raw = uploaded.read()
        content_type = uploaded.mimetype or "image/jpeg"

    if not raw:
        raise ValueError("Provide an image (data URL in JSON or a 'file' upload).")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large (8 MB maximum).")

    try:
        image = Image.open(BytesIO(raw))
        image.load()
        return image, raw, content_type
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc


app = Flask(__name__)
ALLOWED_ORIGIN = os.environ.get("CORS_ALLOW_ORIGIN", "*")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return response


@app.route("/")
def index():
    return jsonify(
        {
            "name": "FaceAttend API",
            "status": "running" if db.get_db() is not None else "unavailable",
            "endpoints": [
                "/api/health", "/api/people", "/api/people/register",
                "/api/attendance", "/api/attendance (POST)",
            ],
        }
    )


@app.route("/api/health")
def health():
    database_ok = db.get_db() is not None
    return jsonify(
        {
            "status": "ok" if database_ok else "error",
            "model_loaded": MODEL is not None,
            "model_error": MODEL_ERROR,
            "database": "connected" if database_ok else "disconnected",
            "database_error": db.db_error(),
            "database_name": db.MONGO_DB,
            "image_storage": "MongoDB GridFS (face_images)",
            "match_threshold": db.MATCH_THRESHOLD,
            "embedding_mode": os.environ.get("FACE_EMBEDDING_MODE", "features"),
            "people_count": len(db.list_people()),
            "event_count": db.event_count(),
        }
    )


@app.route("/api/people", methods=["GET"])
def people_list():
    if db.get_db() is None:
        return jsonify(error=f"Database unavailable. {db.db_error()}"), 503
    people = db.list_people()
    return jsonify({"people": people, "total": len(people)})


@app.route("/api/people/register", methods=["POST", "OPTIONS"])
def register():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    if db.get_db() is None:
        return jsonify(error=f"Database unavailable. {db.db_error()}"), 503

    payload, files = request_data()
    person_id = str(payload.get("id") or payload.get("personId") or "").strip()
    name = str(payload.get("name") or "").strip()
    if not person_id or not name:
        return jsonify(error="Both 'id' and 'name' are required."), 400
    if len(person_id) > 64 or len(name) > 80:
        return jsonify(error="ID or name is too long."), 400

    try:
        image, raw_image, content_type = decode_image(payload, files)
        vector = extract_embedding(image)
        person, is_new = db.register_person(person_id, name, vector, raw_image, content_type)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return jsonify(error=f"Registration failed: {exc}"), 500

    return jsonify(
        {
            "status": "registered" if is_new else "photo_added",
            "person": db.public_person(person),
            "message": (
                f"{name} registered with ID {person_id}."
                if is_new else f"Added another photo for {name}."
            ),
        }
    )


@app.route("/api/people/<person_id>", methods=["DELETE", "OPTIONS"])
def delete_person(person_id: str):
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    if db.get_db() is None:
        return jsonify(error=f"Database unavailable. {db.db_error()}"), 503
    if db.delete_person(person_id):
        return jsonify({"status": "deleted", "id": person_id})
    return jsonify(error="Person not found."), 404


@app.route("/api/photos/<file_id>", methods=["GET"])
def photo(file_id: str):
    """Serve an image stored in MongoDB GridFS."""
    result = db.get_image(file_id)
    if result is None:
        return jsonify(error="Image not found."), 404
    raw, content_type = result
    return Response(raw, mimetype=content_type, headers={"Cache-Control": "public, max-age=3600"})


@app.route("/api/attendance", methods=["GET"])
def attendance_list():
    if db.get_db() is None:
        return jsonify(error=f"Database unavailable. {db.db_error()}"), 503
    try:
        limit = int(request.args.get("limit", 30))
    except (TypeError, ValueError):
        limit = 30
    limit = max(1, min(limit, 200))
    events = db.recent_events(limit)
    return jsonify({"total": len(events), "events": events})


@app.route("/api/attendance", methods=["POST", "OPTIONS"])
def attendance_scan():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    if db.get_db() is None:
        return jsonify(error=f"Database unavailable. {db.db_error()}"), 503
    if not db.list_people():
        return jsonify(error="No one is registered yet. Register a face first."), 400

    payload, files = request_data()
    mode = str(payload.get("mode", "in")).strip().lower()
    if mode not in ("in", "out"):
        return jsonify(error="Mode must be 'in' or 'out'."), 400

    try:
        image, raw_image, content_type = decode_image(payload, files)
        vector = extract_embedding(image)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return jsonify(error=f"Recognition failed: {exc}"), 500

    matched_id = None
    match = db.match_face(vector)
    if match["matched"]:
        matched_id = match["best"]["id"]
    snapshot_id = db.store_image(raw_image, content_type, matched_id or "unknown", "scan")
    snapshot_url = f"/api/photos/{snapshot_id}" if snapshot_id else None

    pkl = pkl_prediction(vector)
    response: Dict[str, Any] = {
        "mode": mode,
        "match": match,
        "snapshot_url": snapshot_url,
        "model_guess": pkl,
        "person": None,
        "event": None,
        "summary": None,
    }

    if not match["matched"]:
        response.update(
            status="no-match",
            message="No registered face matched. Register first or try again in better light.",
        )
        return jsonify(response)

    person = db.find_person(match["best"]["id"])
    person_info = db.public_person(person)
    response["person"] = person_info
    previous = db.last_event(person["id"])

    if mode == "in" and previous and previous.get("type") == "in":
        response.update(
            status="already_in",
            message=f"{person['name']} is already checked in.",
            summary=db.today_summary(person["id"]),
        )
        return jsonify(response)

    if mode == "out" and (not previous or previous.get("type") == "out"):
        response.update(
            status="no_open_checkin",
            message=f"{person['name']} has no open check-in to close.",
            summary=db.today_summary(person["id"]),
        )
        return jsonify(response)

    event = db.insert_event(
        person["id"],
        person["name"],
        mode,
        match["best"]["score"],
        pkl["guess"] if pkl else None,
        pkl["confidence"] if pkl else None,
        snapshot_id,
    )
    response["person"] = db.public_person(db.find_person(person["id"]))
    response.update(
        status="checked_in" if mode == "in" else "checked_out",
        event=db.public_event(event),
        summary=db.today_summary(person["id"]),
        message=(
            f"{person['name']} checked in. Attendance count: {response['person']['attendance_count']}."
            if mode == "in" else f"{person['name']} checked out."
        ),
    )
    return jsonify(response)


@app.errorhandler(404)
def not_found(_error):
    return jsonify(error="Endpoint not found."), 404


@app.errorhandler(405)
def method_not_allowed(_error):
    return jsonify(error="Method not allowed."), 405


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5002"))
    if db.get_db() is None:
        app.logger.warning("MongoDB unavailable: %s", db.db_error())
    if MODEL is None:
        app.logger.warning("Model not loaded: %s", MODEL_ERROR)
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
