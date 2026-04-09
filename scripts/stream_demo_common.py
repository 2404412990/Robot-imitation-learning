import os
import pickle

import cv2
import numpy as np


def env_flag(name, default=True):
    raw = os.environ.get(name, str(int(bool(default))))
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def rotate_frame_bgr(frame, rotate_deg):
    if rotate_deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate_deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotate_deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def sanitize_preview_frame(frame):
    """Make frame contiguous and suppress occasional 1-pixel left-edge artifacts."""
    if frame is None:
        return frame
    frame = np.ascontiguousarray(frame)
    if frame.ndim == 3 and frame.shape[1] > 1:
        frame[:, 0, :] = frame[:, 1, :]
    elif frame.ndim == 2 and frame.shape[1] > 1:
        frame[:, 0] = frame[:, 1]
    return frame


def estimate_fps_from_timestamps(timestamps, fallback_fps=30.0):
    if timestamps is None:
        return float(max(1.0, fallback_fps))
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    ts = ts[np.isfinite(ts)]
    if ts.size < 2:
        return float(max(1.0, fallback_fps))
    ts = np.sort(ts)
    dt = np.diff(ts)
    dt = dt[dt > 1e-4]
    if dt.size == 0:
        return float(max(1.0, fallback_fps))
    est = 1.0 / float(np.median(dt))
    if not np.isfinite(est):
        return float(max(1.0, fallback_fps))
    return float(np.clip(est, 1.0, 120.0))


def init_preview_window(window_name, width, height, max_width, max_height):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    scale = min(max_width / float(width), max_height / float(height), 1.0)
    init_w = max(1, int(width * scale))
    init_h = max(1, int(height * scale))
    cv2.resizeWindow(window_name, init_w, init_h)
    return init_w, init_h


class TailStreamWriter:
    def __init__(self, stream_npz_dir, flush_interval=1):
        self.stream_npz_dir = os.path.abspath(stream_npz_dir)
        self.flush_interval = max(1, int(flush_interval))
        self.tail_records = 0

        os.makedirs(self.stream_npz_dir, exist_ok=True)
        self._cleanup_stale_files()

        self.tail_path = os.path.join(self.stream_npz_dir, "stream_tail.pkl")
        self._f = open(self.tail_path, "wb")

    def _cleanup_stale_files(self):
        for name in os.listdir(self.stream_npz_dir):
            if name.startswith("chunk_") and name.endswith(".npz"):
                try:
                    os.remove(os.path.join(self.stream_npz_dir, name))
                except OSError:
                    pass

        for name in ("stream_done.flag", "stream_tail.pkl"):
            path = os.path.join(self.stream_npz_dir, name)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def write_record(self, frame_id, start_time, params):
        if self._f is None:
            raise RuntimeError("TailStreamWriter is already closed.")
        out = self._f
        pickle.dump(
            {
                "frame_id": int(frame_id),
                "start_time": float(start_time),
                "params": params,
            },
            out,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        self.tail_records += 1
        if (self.tail_records % self.flush_interval) == 0:
            out.flush()

    def close(self, frame_count):
        out = self._f
        if out is None:
            return
        try:
            pickle.dump(
                {
                    "done": True,
                    "frame_count": int(frame_count),
                },
                out,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            out.flush()
        finally:
            out.close()
            self._f = None
