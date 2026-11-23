"""MongoDB storage for registered faces and IN/OUT attendance events."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
from bson import ObjectId
from bson.errors import InvalidId
from gridfs import GridFSBucket
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://127.0.0.1:27017/")
MONGO_DB = os.environ.get("MONGO_DB", "faceattend")
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.70"))

_CLIENT = None
_DB = None
_ERROR: str | None = None


def get_db():
    """Return the database, or None with a readable error if Mongo is down."""
    global _CLIENT, _DB, _ERROR
    if _DB is not None:
        return _DB
    if _ERROR is not None:
        return None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[MONGO_DB]
        db.people.create_index("id", unique=True)
        db.attendance.create_index([("person_id", ASCENDING), ("timestamp", DESCENDING)])
        db.attendance.create_index([("timestamp", DESCENDING)])
        _CLIENT, _DB = client, db
        return db
    except PyMongoError as exc:  # pragma: no cover - connection guard
        _ERROR = str(exc)
        return None


def db_error() -> str | None:
    return _ERROR


def image_bucket() -> GridFSBucket | None:
    """GridFS stores the actual image files (registration + scan snapshots)."""
    database = get_db()
    if database is None:
        return None
    return GridFSBucket(database, bucket_name="face_images")


def store_image(raw: bytes, content_type: str, person_id: str, kind: str) -> Any | None:
    bucket = image_bucket()
    if bucket is None:
        return None
    return bucket.upload_from_stream(
        f"{person_id}_{kind}.jpg",
        BytesIO(raw),
        metadata={
            "person_id": person_id,
            "kind": kind,
            "content_type": content_type or "image/jpeg",
            "uploaded_at": utcnow(),
        },
    )


def get_image(file_id: str) -> tuple[bytes, str] | None:
    bucket = image_bucket()
    if bucket is None:
        return None
    try:
        oid = ObjectId(file_id)
    except (InvalidId, TypeError):
        return None
    database = get_db()
    info = database["face_images.files"].find_one({"_id": oid}) if database is not None else None
    if info is None:
        return None
    output = BytesIO()
    bucket.download_to_stream(oid, output)
    metadata = (info.get("metadata") or {}) if isinstance(info, dict) else {}
    return output.getvalue(), str(metadata.get("content_type") or "image/jpeg")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def iso(value: Any) -> str | None:
    stamp = as_utc(value)
    return stamp.isoformat() if stamp else None


def day_key(stamp: datetime | None = None) -> str:
    return (stamp or utcnow()).astimezone(timezone.utc).strftime("%Y-%m-%d")


def public_person(doc: Dict[str, Any]) -> Dict[str, Any]:
    photo_ids = [str(pid) for pid in (doc.get("photo_ids") or [])]
    return {
        "id": doc.get("id"),
        "name": doc.get("name"),
        "photos": len(photo_ids) if photo_ids else len(doc.get("embeddings") or []),
        "photo_urls": [f"/api/photos/{pid}" for pid in photo_ids],
        "attendance_count": int(doc.get("attendance_count") or 0),
        "created_at": iso(doc.get("created_at")),
        "updated_at": iso(doc.get("updated_at")),
    }


def public_event(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(doc.get("_id", "")),
        "person_id": doc.get("person_id"),
        "name": doc.get("name"),
        "type": doc.get("type"),
        "timestamp": iso(doc.get("timestamp")),
        "date": doc.get("date"),
        "match_score": round(float(doc.get("match_score") or 0), 4),
        "duration_minutes": doc.get("duration_minutes"),
        "snapshot_url": f"/api/photos/{doc['snapshot_id']}" if doc.get("snapshot_id") else None,
    }


def list_people() -> List[Dict[str, Any]]:
    db = get_db()
    if db is None:
        return []
    docs = db.people.find({}, {"_id": 0}).sort("name", ASCENDING)
    return [public_person(doc) for doc in docs]


def find_person(person_id: str) -> Dict[str, Any] | None:
    db = get_db()
    if db is None:
        return None
    return db.people.find_one({"id": person_id})


def register_person(
    person_id: str,
    name: str,
    vector: np.ndarray,
    raw_image: bytes,
    content_type: str,
) -> tuple[Dict[str, Any] | None, bool]:
    """Insert a person or append a photo + embedding to an existing ID."""
    db = get_db()
    if db is None:
        return None, False
    now = utcnow()
    photo_id = store_image(raw_image, content_type, person_id, "registration")
    existing = db.people.find_one({"id": person_id})
    if existing:
        push: Dict[str, Any] = {"embeddings": vector.astype(float).tolist()}
        if photo_id is not None:
            push["photo_ids"] = photo_id
        db.people.update_one(
            {"id": person_id},
            {"$set": {"name": name, "updated_at": now}, "$push": push},
        )
        return find_person(person_id), False
    db.people.insert_one(
        {
            "id": person_id,
            "name": name,
            "embeddings": [vector.astype(float).tolist()],
            "photo_ids": [photo_id] if photo_id else [],
            "attendance_count": 0,
            "created_at": now,
            "updated_at": now,
        }
    )
    return find_person(person_id), True


def delete_person(person_id: str) -> bool:
    db = get_db()
    if db is None:
        return False
    result = db.people.delete_one({"id": person_id})
    if result.deleted_count:
        db.attendance.delete_many({"person_id": person_id})
        bucket = image_bucket()
        if bucket is not None:
            for image in bucket.find({"metadata.person_id": person_id}):
                bucket.delete(image._id)
        return True
    return False


def match_face(vector: np.ndarray, threshold: float | None = None) -> Dict[str, Any]:
    """Nearest registered face by cosine similarity.

    Registration and scanning both use the same extractor, so this works for
    any newly registered ID (unlike the two-class pickle classifier).
    """
    limit = MATCH_THRESHOLD if threshold is None else threshold
    db = get_db()
    if db is None:
        return {"matched": False, "best": None, "candidates": []}

    query = vector.astype(np.float64)
    norm = float(np.linalg.norm(query))
    if norm == 0:
        return {"matched": False, "best": None, "candidates": []}
    query = query / norm

    candidates: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for person in db.people.find({}, {"id": 1, "name": 1, "embeddings": 1}):
        for embedding in person.get("embeddings") or []:
            other = np.asarray(embedding, dtype=np.float64)
            other_norm = float(np.linalg.norm(other))
            if other_norm == 0:
                continue
            score = float(np.dot(query, other / other_norm))
            item = {"id": person["id"], "name": person["name"], "score": score}
            candidates.append(item)
            if best is None or score > best["score"]:
                best = item

    candidates.sort(key=lambda row: row["score"], reverse=True)
    unique: List[Dict[str, Any]] = []
    seen = set()
    for item in candidates:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        unique.append(
            {"id": item["id"], "name": item["name"], "score": round(item["score"], 4)}
        )
        if len(unique) == 3:
            break

    matched = best is not None and best["score"] >= limit
    if best is not None:
        best = dict(best)
        best["score"] = round(best["score"], 4)
    return {"matched": matched, "best": best, "candidates": unique, "threshold": limit}


def last_event(person_id: str, date: str | None = None) -> Dict[str, Any] | None:
    db = get_db()
    if db is None:
        return None
    return db.attendance.find_one(
        {"person_id": person_id, "date": date or day_key()},
        sort=[("timestamp", DESCENDING)],
    )


def insert_event(
    person_id: str,
    name: str,
    event_type: str,
    match_score: float,
    pkl_guess: str | None = None,
    pkl_confidence: float | None = None,
    snapshot_id: Any | None = None,
) -> Dict[str, Any]:
    db = get_db()
    if db is None:
        raise RuntimeError("Database is unavailable.")
    now = utcnow()
    payload: Dict[str, Any] = {
        "person_id": person_id,
        "name": name,
        "type": event_type,
        "timestamp": now,
        "date": day_key(now),
        "match_score": float(match_score),
        "pkl_guess": pkl_guess,
        "pkl_confidence": pkl_confidence,
        "snapshot_id": snapshot_id,
    }
    previous = last_event(person_id, payload["date"])
    if event_type == "out" and previous and previous.get("type") == "in":
        start = as_utc(previous.get("timestamp"))
        if start:
            payload["duration_minutes"] = round((now - start).total_seconds() / 60, 1)

    result = db.attendance.insert_one(payload)
    if event_type == "in":
        db.people.update_one({"id": person_id}, {"$inc": {"attendance_count": 1}})
    payload["_id"] = result.inserted_id
    return payload


def today_summary(person_id: str) -> Dict[str, Any]:
    db = get_db()
    if db is None:
        return {}
    events = list(
        db.attendance.find({"person_id": person_id, "date": day_key()}).sort("timestamp", ASCENDING)
    )
    check_in = next((iso(e["timestamp"]) for e in events if e["type"] == "in"), None)
    check_out = next((iso(e["timestamp"]) for e in reversed(events) if e["type"] == "out"), None)
    duration = sum(float(e.get("duration_minutes") or 0) for e in events if e["type"] == "out")
    return {
        "date": day_key(),
        "check_in": check_in,
        "check_out": check_out,
        "duration_minutes": round(duration, 1),
        "events": len(events),
    }


def recent_events(limit: int = 30) -> List[Dict[str, Any]]:
    db = get_db()
    if db is None:
        return []
    docs = db.attendance.find({}).sort("timestamp", DESCENDING).limit(limit)
    return [public_event(doc) for doc in docs]


def event_count() -> int:
    db = get_db()
    if db is None:
        return 0
    return int(db.attendance.count_documents({}))
