"""Face recognition web service: trusted / blocked / unknown.

POST /verify (multipart field "file") -> exactly one of:
  {"status": "trusted", ...}  - face matched trusted_faces table
  {"status": "blocked", ...}  - face matched blocked_faces table
  {"status": "unknown", "face_id": "...", ...} - in neither table

The "unknown" response contains a face_id. The calling API can later send
that face_id (optionally) to enroll the person:

  POST /trust {"face_id": "...", "name": "Alice"}  -> moves to trusted_faces
  POST /block {"face_id": "...", "name": "...", "reason": "..."} -> blocked_faces

Stack mirrors the Face Detection service: FastAPI + OpenCV DNN
(YuNet detect + SFace embed, no torch needed), SQLite persistence,
10 MB cap, CORS open, /health endpoint.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .recognizer import MATCH_THRESHOLD, FaceRecognizer, cosine_score, decode_image
from .store import FaceStore

log = logging.getLogger("face-recognition")
MAX_BYTES = 10 * 1024 * 1024  # same 10 MB cap as detection service

rec: FaceRecognizer | None = None
store: FaceStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rec, store
    rec = FaceRecognizer()
    rec.warmup()
    store = FaceStore()
    store.cleanup_pending()
    log.info("face recognizer ready (backend=%s)", rec.backend)
    yield


app = FastAPI(title="Face Recognition Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
)


def _require_ready():
    if rec is None or store is None:
        raise HTTPException(503, "recognizer not ready")
    return rec, store


class TrustRequest(BaseModel):
    face_id: str
    name: str = ""


class BlockRequest(BaseModel):
    face_id: str
    name: str = ""
    reason: str = ""


def _best_match(query: np.ndarray, rows: list[tuple[str, int, str, np.ndarray]]):
    """Return (best_trusted, best_blocked) as (row_id, name, score) or None."""
    best_t = None
    best_b = None
    for list_name, row_id, name, emb in rows:
        s = cosine_score(query, emb)
        cand = (row_id, name, round(s, 4))
        if list_name == "trusted":
            if best_t is None or s > best_t[2]:
                best_t = cand
        else:
            if best_b is None or s > best_b[2]:
                best_b = cand
    return best_t, best_b


@app.get("/", response_class=JSONResponse)
def root():
    return {
        "service": "face-recognition",
        "usage": "POST /verify with multipart field 'file'. Returns {status: trusted|blocked|unknown}.",
        "endpoints": ["/verify", "/trust", "/block", "/enroll",
                      "/trusted", "/blocked", "/stats", "/health"],
    }


@app.get("/health")
def health():
    counts = store.stats() if store else {"trusted": 0, "blocked": 0, "pending": 0}
    return {"status": "ok",
            "backend": rec.backend if rec else "loading",
            "recognition_ready": rec.recognition_ready if rec else False,
            **counts}


@app.get("/stats")
def stats():
    _, st = _require_ready()
    assert st is not None
    return st.stats()


@app.post("/verify")
async def verify(
    file: UploadFile = File(..., description="Image file (jpg/png/webp/...)"),
    threshold: float = Query(MATCH_THRESHOLD, ge=0.1, le=0.99,
                             description="Cosine similarity cutoff. Higher = stricter."),
    include_faces: bool = Query(False, description="If true, include face boxes."),
):
    recognizer, st = _require_ready()
    assert recognizer is not None and st is not None

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file too large (>{MAX_BYTES // 1024 // 1024} MB)")

    img = decode_image(data)
    if img is None:
        raise HTTPException(400, "undecodable image: upload a valid jpg/png/webp")

    t0 = time.perf_counter()
    faces, det_ms = recognizer.detect(img)

    payload: dict = {
        "face_count": len(faces),
        "backend": recognizer.backend,
    }
    if include_faces:
        payload["faces"] = [
            {k: f[k] for k in ("x", "y", "w", "h", "score")} for f in faces
        ]

    if not faces:
        payload.update({"status": "unknown", "face_id": None,
                        "detail": "no face found",
                        "time_ms": round((time.perf_counter() - t0) * 1000.0, 2)})
        return payload

    if not recognizer.recognition_ready:
        payload.update({"status": "unknown", "face_id": None,
                        "detail": "recognition model unavailable (detection-only fallback)",
                        "time_ms": round((time.perf_counter() - t0) * 1000.0, 2)})
        return payload

    # Largest face drives the verdict (documented access-control semantic).
    primary = max(faces, key=lambda f: f["w"] * f["h"])
    if primary["_raw"] is None:
        payload.update({"status": "unknown", "face_id": None,
                        "detail": "detection fallback has no embeddings",
                        "time_ms": round((time.perf_counter() - t0) * 1000.0, 2)})
        return payload

    # alignCrop needs the SAME (downscaled) image detect() ran on.
    small, _ = recognizer._preprocess(img)
    query = recognizer.embed(small, primary["_raw"])
    if query is None:
        payload.update({"status": "unknown", "face_id": None,
                        "detail": "embedding failed",
                        "time_ms": round((time.perf_counter() - t0) * 1000.0, 2)})
        return payload

    rows = st.all_embeddings()
    best_t, best_b = _best_match(query, rows)
    tb = best_t[2] if best_t else 0.0
    bb = best_b[2] if best_b else 0.0

    # Blocked wins ties (secure default when a face is in both tables).
    if best_b is not None and bb >= threshold and bb >= tb:
        payload.update({"status": "blocked",
                        "match": {"list": "blocked", "id": best_b[0],
                                  "name": best_b[1], "score": best_b[2]},
                        "time_ms": round((time.perf_counter() - t0) * 1000.0, 2)})
        return payload
    if best_t is not None and tb >= threshold:
        payload.update({"status": "trusted",
                        "match": {"list": "trusted", "id": best_t[0],
                                  "name": best_t[1], "score": best_t[2]},
                        "time_ms": round((time.perf_counter() - t0) * 1000.0, 2)})
        return payload

    face_id = uuid.uuid4().hex
    st.add_pending(face_id, query)
    payload.update({"status": "unknown", "face_id": face_id,
                    "detail": f"no match above threshold {threshold}",
                    "closest_trusted_score": round(tb, 4),
                    "closest_blocked_score": round(bb, 4),
                    "time_ms": round((time.perf_counter() - t0) * 1000.0, 2)})
    return payload


@app.post("/trust")
def trust(req: TrustRequest):
    _, st = _require_ready()
    assert st is not None
    emb = st.take_pending(req.face_id)
    if emb is None:
        raise HTTPException(404, "unknown face_id (expired or never seen). POST /verify first.")
    name = req.name or f"person-{req.face_id[:8]}"
    row_id = st.add_trusted(emb, name)
    return {"status": "ok", "list": "trusted", "id": row_id, "name": name}


@app.post("/block")
def block(req: BlockRequest):
    _, st = _require_ready()
    assert st is not None
    emb = st.take_pending(req.face_id)
    if emb is None:
        raise HTTPException(404, "unknown face_id (expired or never seen). POST /verify first.")
    name = req.name or f"person-{req.face_id[:8]}"
    row_id = st.add_blocked(emb, name, req.reason)
    return {"status": "ok", "list": "blocked", "id": row_id, "name": name}


@app.post("/enroll")
async def enroll(
    file: UploadFile = File(..., description="Image file with a single face"),
    list: Literal["trusted", "blocked"] = Form(..., description="Target table"),
    name: str = Form("", description="Person label"),
    reason: str = Form("", description="Block reason (blocked only)"),
):
    """Direct image -> table enrollment (skips the unknown/face_id round-trip)."""
    recognizer, st = _require_ready()
    assert recognizer is not None and st is not None
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    img = decode_image(data)
    if img is None:
        raise HTTPException(400, "undecodable image")
    faces, _ = recognizer.detect(img)
    if not faces:
        raise HTTPException(400, "no face found in image")
    if not recognizer.recognition_ready:
        raise HTTPException(503, "recognition model unavailable")
    primary = max(faces, key=lambda f: f["w"] * f["h"])
    if primary["_raw"] is None:
        raise HTTPException(503, "detection fallback has no embeddings")
    small, _ = recognizer._preprocess(img)
    query = recognizer.embed(small, primary["_raw"])
    if query is None:
        raise HTTPException(500, "embedding failed")
    label = name or "person"
    if list == "trusted":
        row_id = st.add_trusted(query, label)
    else:
        row_id = st.add_blocked(query, label, reason)
    return {"status": "ok", "list": list, "id": row_id, "name": label}


@app.get("/trusted")
def list_trusted():
    _, st = _require_ready()
    assert st is not None
    return {"trusted": st.list_trusted()}


@app.get("/blocked")
def list_blocked():
    _, st = _require_ready()
    assert st is not None
    return {"blocked": st.list_blocked()}


@app.delete("/trusted/{row_id}")
def delete_trusted(row_id: int):
    _, st = _require_ready()
    assert st is not None
    if not st.delete_trusted(row_id):
        raise HTTPException(404, "trusted id not found")
    return {"status": "ok", "deleted": row_id}


@app.delete("/blocked/{row_id}")
def delete_blocked(row_id: int):
    _, st = _require_ready()
    assert st is not None
    if not st.delete_blocked(row_id):
        raise HTTPException(404, "blocked id not found")
    return {"status": "ok", "deleted": row_id}
