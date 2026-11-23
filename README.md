# FaceAttend (MongoDB)

React UI + Flask API for face-based IN / OUT attendance. Registered faces (ID,
name, image embeddings) are stored in **MongoDB**. A scan finds the closest
registered face; on a match the event is saved with a timestamp and the
`attendance_count` increases.

> বাংলা: প্রথমে ID + নাম + ছবি দিয়ে.register করবেন। পরে ছবি স্ক্যান করলে MongoDB তে খুঁজে মিললে attendance count বাড়বে। IN এবং OUT — দুইটার জন্য আলাদা করে face scan + সময় save হবে।

## Flow

1. **Register** — enter ID + name, capture/upload a photo. Embedding is stored in
   MongoDB. Add 2–3 photos with the same ID for better accuracy.
2. **Check IN** — scan face → nearest registered face is found by cosine
   similarity → `in` event saved, `attendance_count + 1`.
   Already checked in? It reports `already_in` instead of a duplicate.
3. **Check OUT** — scan face → `out` event saved with time and duration since
   check-in. Without an open check-in it reports `no_open_checkin`.

The supplied `model/attendance_model.pkl` (`SVC` + `LabelEncoder`, 1280-d,
`person_A` / `person_B`) is still loaded and its guess is shown as a secondary
signal. Identity matching is done against MongoDB so any new ID works.

## MongoDB setup

Local MongoDB is used by default (`mongodb://127.0.0.1:27017/`, database
`faceattend`). Override with environment variables:

```powershell
$env:MONGO_URI = "mongodb://127.0.0.1:27017/"
$env:MONGO_DB = "faceattend"
```

Collections are created automatically:

- `people` — `{ id, name, embeddings: [[1280 floats], ...], photo_ids: [...], attendance_count, ... }`
- `attendance` — `{ person_id, name, type: "in"|"out", timestamp, date, match_score, duration_minutes, snapshot_id }`
- `face_images` (GridFS) — actual image files for registration photos and scan snapshots

Images are served back through `GET /api/photos/<file_id>` and shown as thumbnails in the UI.

## Run it

```powershell
# Terminal 1 — API (http://localhost:5002)
cd backend
pip install -r requirements.txt
python app.py
```

```powershell
# Terminal 2 — UI (http://localhost:5173)
cd frontend
npm install
npm run dev
```

## Matching config

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATCH_THRESHOLD` | `0.70` | Minimum cosine similarity to accept a match |
| `FACE_EMBEDDING_MODE` | `features` | `features` (MobileNetV2 1280-d) or `pixels` (40×32 grayscale) |
| `MONGO_URI` / `MONGO_DB` | local / `faceattend` | MongoDB connection |
| `PORT` | `5002` | Flask port (5001 is used by another project here) |

If faces never match, lower `MATCH_THRESHOLD` (e.g. `0.6`) or try the other
embedding mode. If you have the original training preprocessing, replace
`extract_embedding()` in `backend/embeddings.py`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | DB + model status, counts, config |
| GET | `/api/people` | registered people with counts |
| POST | `/api/people/register` | `{ id, name, image }` or multipart `id`, `name`, `file` |
| DELETE | `/api/people/<id>` | delete a person, their events and images |
| GET | `/api/photos/<file_id>` | serve a stored image from GridFS |
| POST | `/api/attendance` | `{ image, mode: "in"\|"out" }` or multipart `file`, `mode` |
| GET | `/api/attendance?limit=30` | latest IN/OUT events |

## Layout

```
faceAttend/
├── model/attendance_model.pkl
├── backend/
│   ├── app.py            # Flask API
│   ├── database.py       # MongoDB storage, matching, IN/OUT logic
│   ├── embeddings.py     # image -> 1280-d vector
│   └── requirements.txt
└── frontend/
    └── src/App.jsx       # register + IN/OUT UI
```
