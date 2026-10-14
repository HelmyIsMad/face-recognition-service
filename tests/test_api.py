from contextlib import contextmanager

import cv2
import numpy as np
from fastapi.testclient import TestClient

import app.main as main
from app.store import FaceStore


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


def _rand_emb(rng: np.random.Generator | None = None) -> np.ndarray:
    """Random unit embedding (normal dist -> cosine ~0 between identities)."""
    rng = rng if rng is not None else np.random.default_rng()
    v = rng.standard_normal(128).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


@contextmanager
def _client(tmp_path, monkeypatch):
    # NOTE: patch AFTER lifespan runs — lifespan reassigns main.store on entry,
    # so patching before TestClient() gets overwritten (leaks into real DB).
    with TestClient(main.app) as client:
        monkeypatch.setattr(main, "store", FaceStore(tmp_path / "test.db"))
        yield client


def test_health(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "backend" in body


def test_blank_image_is_unknown_no_face(tmp_path, monkeypatch):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with _client(tmp_path, monkeypatch) as client:
        r = client.post("/verify", files={"file": ("blank.jpg", _encode(blank), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "unknown"
    assert body["face_count"] == 0
    assert body["face_id"] is None


def test_invalid_image_rejected(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        r = client.post("/verify", files={"file": ("x.jpg", b"not-an-image", "image/jpeg")})
    assert r.status_code == 400


def test_trust_unknown_face_id_404(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        r = client.post("/trust", json={"face_id": "nope", "name": "Ghost"})
    assert r.status_code == 404


def test_block_unknown_face_id_404(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        r = client.post("/block", json={"face_id": "nope"})
    assert r.status_code == 404


def test_pending_trust_flow(tmp_path, monkeypatch):
    emb = _rand_emb()
    with _client(tmp_path, monkeypatch) as client:
        main.store.add_pending("fid123", emb)
        r = client.post("/trust", json={"face_id": "fid123", "name": "Alice"})
        assert r.status_code == 200, r.text
        assert r.json()["list"] == "trusted"
        row_id = r.json()["id"]
        # second use of same face_id must fail (consumed)
        r2 = client.post("/trust", json={"face_id": "fid123"})
        assert r2.status_code == 404
        # listed, then deletable
        assert any(p["id"] == row_id for p in client.get("/trusted").json()["trusted"])
        assert client.delete(f"/trusted/{row_id}").status_code == 200
        assert client.delete(f"/trusted/{row_id}").status_code == 404


def test_verify_matches_seeded_embedding(tmp_path, monkeypatch):
    """Mock detect+embed so no real face photo is needed."""
    emb = _rand_emb()
    fake_face = {"x": 10, "y": 10, "w": 100, "h": 100, "score": 0.99,
                 "_raw": np.zeros(15, dtype=np.float32)}
    blank = np.zeros((320, 320, 3), dtype=np.uint8)

    with _client(tmp_path, monkeypatch) as client:
        main.store.add_trusted(emb, "Bob")
        monkeypatch.setattr(main.rec, "detect", lambda img: ([fake_face], 1.0))
        monkeypatch.setattr(main.rec, "embed", lambda small, raw: emb.copy())
        monkeypatch.setattr(main.rec, "_preprocess",
                            lambda img: (img, 1.0))
        r = client.post("/verify", files={"file": ("b.jpg", _encode(blank), "image/jpeg")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "trusted"
        assert body["match"]["name"] == "Bob"
        assert body["match"]["score"] >= 0.99


def test_blocked_wins_tie(tmp_path, monkeypatch):
    emb = _rand_emb()
    fake_face = {"x": 10, "y": 10, "w": 100, "h": 100, "score": 0.99,
                 "_raw": np.zeros(15, dtype=np.float32)}
    blank = np.zeros((320, 320, 3), dtype=np.uint8)

    with _client(tmp_path, monkeypatch) as client:
        main.store.add_trusted(emb, "Double")
        main.store.add_blocked(emb, "Double", "test")
        monkeypatch.setattr(main.rec, "detect", lambda img: ([fake_face], 1.0))
        monkeypatch.setattr(main.rec, "embed", lambda small, raw: emb.copy())
        monkeypatch.setattr(main.rec, "_preprocess",
                            lambda img: (img, 1.0))
        r = client.post("/verify", files={"file": ("b.jpg", _encode(blank), "image/jpeg")})
        assert r.json()["status"] == "blocked"


def test_unknown_returns_face_id_and_enrolls(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    emb = _rand_emb(rng)
    other = _rand_emb(rng)  # independent identity -> cosine ~0, below threshold
    fake_face = {"x": 10, "y": 10, "w": 100, "h": 100, "score": 0.99,
                 "_raw": np.zeros(15, dtype=np.float32)}
    blank = np.zeros((320, 320, 3), dtype=np.uint8)

    with _client(tmp_path, monkeypatch) as client:
        main.store.add_trusted(other, "Stranger")
        monkeypatch.setattr(main.rec, "detect", lambda img: ([fake_face], 1.0))
        monkeypatch.setattr(main.rec, "embed", lambda small, raw: emb.copy())
        monkeypatch.setattr(main.rec, "_preprocess",
                            lambda img: (img, 1.0))
        r = client.post("/verify", files={"file": ("b.jpg", _encode(blank), "image/jpeg")})
        assert r.json()["status"] == "unknown"
        fid = r.json()["face_id"]
        assert fid
        # enroll via face_id round-trip
        r2 = client.post("/block", json={"face_id": fid, "name": "Intruder"})
        assert r2.status_code == 200
        # same face now blocked
        r3 = client.post("/verify", files={"file": ("b.jpg", _encode(blank), "image/jpeg")})
        assert r3.json()["status"] == "blocked"
