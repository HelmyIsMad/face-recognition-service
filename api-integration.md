# API Integration Guide — Face Recognition Service

Base URL (local): `http://localhost:8001`
Interactive docs: `http://localhost:8001/docs`

Answers: **is this face trusted, blocked, or unknown?**

## 1. Start the server

```powershell
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

First run downloads YuNet + SFace weights into `models/`. Health check:

```powershell
curl http://localhost:8001/health
# {"status":"ok","backend":"yunet+sface","recognition_ready":true,"trusted":0,"blocked":0,"pending":0}
```

## 2. Endpoints

| Method   | Path             | Purpose                                                      |
|----------|------------------|--------------------------------------------------------------|
| `GET`    | `/`              | Usage info                                                   |
| `GET`    | `/health`        | Liveness + counts                                            |
| `GET`    | `/stats`         | `{trusted, blocked, pending}`                                |
| `POST`   | `/verify`        | Image → `trusted` / `blocked` / `unknown` (+`face_id`)       |
| `POST`   | `/trust`         | JSON `face_id` → enroll into trusted                         |
| `POST`   | `/block`         | JSON `face_id` → enroll into blocked                         |
| `POST`   | `/enroll`        | Image + `list` directly into a table (no `face_id` needed)   |
| `GET`    | `/trusted`       | List trusted entries                                         |
| `GET`    | `/blocked`       | List blocked entries                                         |
| `DELETE` | `/trusted/{id}`  | Delete                                                       |
| `DELETE` | `/blocked/{id}`  | Delete                                                       |

### `POST /verify` — request

- Content-Type: `multipart/form-data`, field **`file`** (jpg/png/webp/bmp, ≤10 MB)
- Query params: `threshold` (default `0.5`, range `0.1`–`0.99`),
  `include_faces` (`true` adds boxes in original-image pixels)

### `POST /verify` — responses (all `200`)

```jsonc
// trusted
{"status": "trusted", "match": {"list": "trusted", "id": 1, "name": "Alice", "score": 0.82}, "face_count": 1, "time_ms": 18.2, "backend": "yunet+sface"}
// blocked
{"status": "blocked", "match": {"list": "blocked", "id": 7, "name": "Bob", "score": 0.77}, "face_count": 1, "time_ms": 17.9, "backend": "yunet+sface"}
// unknown — keep face_id for the optional next step
{"status": "unknown", "face_id": "9f3c...", "face_count": 1, "detail": "no match above threshold 0.5", "closest_trusted_score": 0.31, "closest_blocked_score": 0.22, "time_ms": 18.2, "backend": "yunet+sface"}
// no face found — normal answer, not an error
{"status": "unknown", "face_id": null, "face_count": 0, "detail": "no face found", "time_ms": 9.1, "backend": "yunet+sface"}
```

Status codes: `200` (even for `unknown`), `400` empty/undecodable/no-face-on-`/enroll`,
`404` stale `face_id`, `413` >10 MB, `503` still loading.

### `POST /trust` / `POST /block` — request (JSON)

```jsonc
// /trust
{"face_id": "9f3c...", "name": "Alice"}        // name optional -> person-<id8>
// /block
{"face_id": "9f3c...", "name": "Bob", "reason": "intruder"}
```

Success: `{"status":"ok","list":"trusted","id":1,"name":"Alice"}`.
`face_id` is single-use, expires after 24 h.

## 3. Examples

### cURL — full unknown → trust loop

```bash
# 1. check
curl -F "file=@photo.jpg" http://localhost:8001/verify
# {"status":"unknown","face_id":"9f3c...","face_count":1,...}

# 2. (optionally) enroll
curl -X POST http://localhost:8001/trust \
  -H "Content-Type: application/json" \
  -d '{"face_id":"9f3c...","name":"Alice"}'

# 3. same photo now verifies trusted
curl -F "file=@photo.jpg" http://localhost:8001/verify
# {"status":"trusted","match":{"name":"Alice",...}}
```

### PowerShell

```powershell
Invoke-RestMethod -Uri "http://localhost:8001/verify" -Method Post `
  -Form @{ file = Get-Item "photo.jpg" }

Invoke-RestMethod -Uri "http://localhost:8001/trust" -Method Post `
  -ContentType "application/json" `
  -Body '{"face_id":"9f3c...","name":"Alice"}'
```

### Python (`requests`) — recommended caller pattern

```python
import requests

BASE = "http://localhost:8001"

def check_face(path: str) -> dict:
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/verify", files={"file": f}, timeout=30)
    r.raise_for_status()
    return r.json()

res = check_face("photo.jpg")
if res["status"] == "trusted":
    print("open door for", res["match"]["name"])
elif res["status"] == "blocked":
    print("deny + alarm")
else:
    fid = res["face_id"]
    if fid:  # optionally enroll after human review
        requests.post(f"{BASE}/trust", json={"face_id": fid, "name": "Alice"}, timeout=30)
```

### JavaScript — browser

```html
<input type="file" id="picker" accept="image/*" />
<script>
  picker.onchange = async () => {
    const form = new FormData();
    form.append("file", picker.files[0]);
    const res = await fetch("http://localhost:8001/verify", { method: "POST", body: form });
    const data = await res.json();
    // data.status: "trusted" | "blocked" | "unknown"; data.face_id when unknown
    alert(data.status);
  };
</script>
```

## 4. Notes for integrators

- CORS is open (`*`) — call directly from browsers.
- Send original file bytes; no base64.
- Largest face drives the verdict when several faces are visible.
- Blocked wins ties when a face matches both tables.
- `threshold` trades false accepts vs rejects; start with default `0.5`.
- Poll `GET /health` for load-balancer checks.
