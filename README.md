# Face Recognition Service

Receives an image, returns exactly one of:

- `trusted` — face matched the **trusted** table
- `blocked` — face matched the **blocked** table
- `unknown` — in neither table. Response also contains a `face_id` the
  caller can send back (optionally) to `POST /trust` or `POST /block`.

Built the same way as `Face Detection`:
**FastAPI + OpenCV DNN, no torch needed.**
Detection = YuNet (~5–20 ms CPU), recognition = SFace embedding
(`face_recognition_sface_2021dec.onnx`), cosine similarity.
Models auto-download into `models/` on first run.
Persistence = SQLite (`data/faces.db`) with `trusted_faces`,
`blocked_faces`, `pending_faces` tables.

## Run

```powershell
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

First run downloads YuNet (~300 KB) + SFace (~38 MB) into `models/`.
Uses port **8001** so it can run alongside the detection service on 8000.

## API

Machine-readable contract: [`openapi.json`](openapi.json) (OpenAPI 3.1,
generated from the app — also served live at `/docs`).

### `POST /verify` — multipart field `file`

```powershell
curl -F "file=@photo.jpg" http://localhost:8001/verify
```

Trusted:

```json
{"status": "trusted", "match": {"list": "trusted", "id": 1, "name": "Alice", "score": 0.82}, "face_count": 1, "time_ms": 18.2, "backend": "yunet+sface"}
```

Blocked: same shape with `"status": "blocked"`.

Unknown (note `face_id`):

```json
{"status": "unknown", "face_id": "9f3c...", "face_count": 1, "detail": "no match above threshold 0.5", "closest_trusted_score": 0.31, "closest_blocked_score": 0.22, "time_ms": 18.2, "backend": "yunet+sface"}
```

No face in image → `{"status": "unknown", "face_id": null, "face_count": 0, ...}`.

| Query param     | Default | Meaning                                           |
|-----------------|---------|---------------------------------------------------|
| `threshold`     | `0.5`   | Cosine cutoff 0.1–0.99. Higher = stricter.        |
| `include_faces` | `false` | Adds `faces: [{x,y,w,h,score}]` (original pixels) |

Largest face drives the verdict when several are visible.
Blocked wins ties (secure default).

### Enroll the unknown face

```powershell
curl -X POST http://localhost:8001/trust -H "Content-Type: application/json" -d '{"face_id":"9f3c...","name":"Alice"}'
# {"status":"ok","list":"trusted","id":1,"name":"Alice"}

curl -X POST http://localhost:8001/block -H "Content-Type: application/json" -d '{"face_id":"9f3c...","name":"Bob","reason":"intruder"}'
```

`face_id` is single-use and expires after 24 h (`404` when stale —
`POST /verify` again for a fresh one).

### Direct enroll from image (skips the round-trip)

```powershell
curl -F "file=@alice.jpg" -F "list=trusted" -F "name=Alice" http://localhost:8001/enroll
curl -F "file=@bob.jpg" -F "list=blocked" -F "name=Bob" -F "reason=intruder" http://localhost:8001/enroll
```

### Admin

| Method   | Path             | Purpose                          |
|----------|------------------|----------------------------------|
| `GET`    | `/health`        | `{"status":"ok","backend":...}` + counts |
| `GET`    | `/stats`         | `{trusted, blocked, pending}` counts |
| `GET`    | `/trusted`       | List trusted (id, name, no embeddings) |
| `GET`    | `/blocked`       | List blocked                     |
| `DELETE` | `/trusted/{id}`  | Remove entry                     |
| `DELETE` | `/blocked/{id}`  | Remove entry                     |

## Notes

- 10 MB upload cap, invalid images → `400`, unknown `face_id` → `404`,
  detector loading → `503`.
- Images downscaled to max side 640 px (same as detection service).
- If SFace can't load, `backend` is `haar`/`yunet` and `/verify`
  returns `unknown` (detection-only fallback).

## Docker

```powershell
docker build -t face-recognition .
docker run -p 8001:8000 -v ${PWD}/data:/srv/data face-recognition
```

Mount `data/` or the SQLite DB is lost when the container exits.

## Tests

```powershell
pytest -q
```
