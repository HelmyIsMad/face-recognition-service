"""YuNet detection + SFace embedding.

Same philosophy as the Face Detection service:
  - YuNet via OpenCV DNN for fast CPU face boxes (~5-20 ms)
  - SFace via OpenCV DNN for embeddings (no torch/onnxruntime needed)
  - Single small models auto-downloaded into models/
  - Haar fallback for detection if YuNet can't load
    (recognition then unavailable -> backend="haar", verify returns unknown)

Matching uses cosine similarity (higher = more similar, range 0..1).
Default threshold 0.5 works well for SFace; expose per-request override.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
YUNET_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"

SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
SFACE_PATH = Path(__file__).resolve().parent.parent / "models" / "face_recognition_sface_2021dec.onnx"

SCORE_THRESHOLD = 0.6   # detection strictness (same as detection service)
NMS_THRESHOLD = 0.3
TOP_K = 20
MAX_SIDE = 640          # downscale large images for speed
MATCH_THRESHOLD = 0.5   # cosine similarity cutoff for trusted/blocked


def ensure_model(path: Path, url: str, min_bytes: int = 100_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > min_bytes:
        return path
    print(f"[recognizer] downloading model -> {path}")
    urllib.request.urlretrieve(url, str(path))
    return path


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class FaceRecognizer:
    """Thread-safe YuNet + SFace wrapper."""

    def __init__(
        self,
        yunet_path: Path | None = None,
        sface_path: Path | None = None,
        score_threshold: float = SCORE_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        top_k: int = TOP_K,
        max_side: int = MAX_SIDE,
    ):
        self.score_threshold = score_threshold
        self.max_side = max_side
        self._lock = threading.Lock()
        self._detector = None
        self._haar = None
        self._recognizer = None
        self.backend = "none"

        yunet_path = yunet_path or YUNET_PATH
        try:
            ensure_model(yunet_path, YUNET_URL)
            self._detector = cv2.FaceDetectorYN.create(
                str(yunet_path),
                "",
                (320, 320),
                score_threshold,
                nms_threshold,
                top_k,
            )
            self.backend = "yunet"
        except Exception as exc:
            print(f"[recognizer] YuNet init failed ({exc}), Haar fallback")
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(haar_path)
            self.backend = "haar"

        # SFace needs YuNet-style aligned faces; only init when YuNet is up.
        if self._detector is not None:
            try:
                sface_path = sface_path or SFACE_PATH
                ensure_model(sface_path, SFACE_URL, min_bytes=1_000_000)
                self._recognizer = cv2.FaceRecognizerSF.create(
                    str(sface_path), "", 0, 0  # backend/target = CPU
                )
                self.backend = "yunet+sface"
            except Exception as exc:
                print(f"[recognizer] SFace init failed ({exc})")
                self._recognizer = None

    @property
    def recognition_ready(self) -> bool:
        return self._detector is not None and self._recognizer is not None

    def warmup(self) -> None:
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        try:
            faces, _ = self.detect(dummy)
            if faces and self._recognizer is not None and "_raw" in faces[0]:
                self.embed(dummy, faces[0]["_raw"])
        except Exception:
            pass

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        m = max(h, w)
        if m <= self.max_side:
            return img, 1.0
        scale = self.max_side / float(m)
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return small, scale

    def detect(self, img: np.ndarray) -> tuple[list[dict], float]:
        """Return (faces, ms). Each face: {x,y,w,h,score,_raw} in ORIGINAL coords.

        `_raw` is the YuNet row in SMALL-image coords, kept for SFace alignCrop.
        Haar fallback faces have `_raw=None` (not embeddable).
        """
        t0 = time.perf_counter()
        if img is None or img.size == 0:
            return [], 0.0
        small, scale = self._preprocess(img)
        h, w = small.shape[:2]
        faces: list[dict] = []
        with self._lock:
            if self._detector is not None:
                self._detector.setInputSize((w, h))
                _, results = self._detector.detect(small)
                if results is not None:
                    inv = 1.0 / scale
                    for r in results:
                        x, y, ww, hh = (float(v) * inv for v in r[:4])
                        faces.append({
                            "x": int(x), "y": int(y),
                            "w": int(ww), "h": int(hh),
                            "score": float(r[-1]),
                            "_raw": r.copy(),
                        })
            elif self._haar is not None:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                rects = self._haar.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                inv = 1.0 / scale
                for (x, y, ww, hh) in rects:
                    faces.append({"x": int(x * inv), "y": int(y * inv),
                                  "w": int(ww * inv), "h": int(hh * inv),
                                  "score": 1.0, "_raw": None})
        ms = (time.perf_counter() - t0) * 1000.0
        return faces, ms

    def embed(self, img: np.ndarray, raw_face: np.ndarray) -> np.ndarray | None:
        """SFace embedding for one YuNet detection row. Returns (128,) float32 or None."""
        if self._recognizer is None or raw_face is None:
            return None
        with self._lock:
            try:
                aligned = self._recognizer.alignCrop(img, raw_face)
                feat = self._recognizer.feature(aligned)
                return np.array(feat, dtype=np.float32).ravel()
            except Exception:
                return None

    def embed_largest(self, img: np.ndarray, face_rows: list[np.ndarray] | None = None,
                      small_img: np.ndarray | None = None) -> np.ndarray | None:
        """Helper when caller already ran detect on a (possibly downscaled) image."""
        # NOTE: alignCrop must use the same image coords as detection ran on.
        # detect() runs on the downscaled image, so embed against that.
        return None  # kept for API clarity; main.py embeds explicitly


def decode_image(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
