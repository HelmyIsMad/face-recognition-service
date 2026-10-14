"""SQLite store: trusted / blocked / pending (unknown) faces.

Two persistent tables required by the spec:
  - trusted_faces
  - blocked_faces

Plus a pending_faces table holding embeddings of "unknown" verdicts,
keyed by the face_id returned to the caller. The caller later sends
that face_id to POST /trust or POST /block to enroll the person.
Pending rows expire after PENDING_TTL_S (default 24 h).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "faces.db"
PENDING_TTL_S = 24 * 3600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trusted_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL DEFAULT '',
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS blocked_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_faces (
    face_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL
);
"""


def _to_blob(emb: np.ndarray) -> bytes:
    return np.ascontiguousarray(emb, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class FaceStore:
    def __init__(self, path: Path | None = None, pending_ttl_s: int = PENDING_TTL_S):
        self.path = Path(path) if path else DB_PATH
        self.pending_ttl_s = pending_ttl_s
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- enroll --
    def add_trusted(self, embedding: np.ndarray, name: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO trusted_faces (name, embedding, created_at) VALUES (?, ?, ?)",
                (name, _to_blob(embedding), time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def add_blocked(self, embedding: np.ndarray, name: str = "", reason: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO blocked_faces (name, reason, embedding, created_at) VALUES (?, ?, ?, ?)",
                (name, reason, _to_blob(embedding), time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    # -- lookup for matching --
    def all_embeddings(self) -> list[tuple[str, int, str, np.ndarray]]:
        """Return [(list_name, row_id, name, embedding), ...]."""
        out: list[tuple[str, int, str, np.ndarray]] = []
        with self._lock:
            for row in self._conn.execute("SELECT id, name, embedding FROM trusted_faces"):
                out.append(("trusted", int(row[0]), str(row[1]), _from_blob(row[2])))
            for row in self._conn.execute("SELECT id, name, embedding FROM blocked_faces"):
                out.append(("blocked", int(row[0]), str(row[1]), _from_blob(row[2])))
        return out

    # -- pending (unknown face_ids) --
    def add_pending(self, face_id: str, embedding: np.ndarray) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO pending_faces (face_id, embedding, created_at) VALUES (?, ?, ?)",
                (face_id, _to_blob(embedding), time.time()),
            )
            self._conn.commit()

    def take_pending(self, face_id: str) -> np.ndarray | None:
        """Fetch + delete a pending embedding. Returns None if missing/expired."""
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding, created_at FROM pending_faces WHERE face_id = ?", (face_id,)
            ).fetchone()
            if row is None:
                return None
            blob, created = row
            if time.time() - float(created) > self.pending_ttl_s:
                self._conn.execute("DELETE FROM pending_faces WHERE face_id = ?", (face_id,))
                self._conn.commit()
                return None
            self._conn.execute("DELETE FROM pending_faces WHERE face_id = ?", (face_id,))
            self._conn.commit()
            return _from_blob(blob)

    def cleanup_pending(self) -> int:
        cutoff = time.time() - self.pending_ttl_s
        with self._lock:
            cur = self._conn.execute("DELETE FROM pending_faces WHERE created_at < ?", (cutoff,))
            self._conn.commit()
            return cur.rowcount

    # -- admin --
    def list_trusted(self) -> list[dict]:
        with self._lock:
            return [
                {"id": r[0], "name": r[1], "created_at": r[2]}
                for r in self._conn.execute(
                    "SELECT id, name, created_at FROM trusted_faces ORDER BY id"
                ).fetchall()
            ]

    def list_blocked(self) -> list[dict]:
        with self._lock:
            return [
                {"id": r[0], "name": r[1], "reason": r[2], "created_at": r[3]}
                for r in self._conn.execute(
                    "SELECT id, name, reason, created_at FROM blocked_faces ORDER BY id"
                ).fetchall()
            ]

    def delete_trusted(self, row_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM trusted_faces WHERE id = ?", (row_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def delete_blocked(self, row_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM blocked_faces WHERE id = ?", (row_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def stats(self) -> dict:
        with self._lock:
            t = self._conn.execute("SELECT COUNT(*) FROM trusted_faces").fetchone()[0]
            b = self._conn.execute("SELECT COUNT(*) FROM blocked_faces").fetchone()[0]
            p = self._conn.execute("SELECT COUNT(*) FROM pending_faces").fetchone()[0]
        return {"trusted": t, "blocked": b, "pending": p}
