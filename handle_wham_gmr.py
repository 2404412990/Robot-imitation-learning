import sys
import argparse
import time
import os
import cv2
import csv
import json
import smplx
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES
import socket, struct, threading

import pathlib
HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from scripts.smplx_to_robot_stream import OnlineQposPostprocessor

SMPLX_FOLDER = HERE / "assets" / "body_models"

EXPECTED_CSV_COLUMNS_BY_ROBOT = {
    "unitree_g1": 36,
    "unitree_h1": 26,
    "x02lite": 25,
    "openloong": 38,
}

X02LITE_HEALTH_LIMITS = {
    "L_shoulder_pitch": (-1.5708, 3.14159),
    "L_shoulder_roll": (0.0, 3.14159),
    "L_shoulder_yaw": (-2.26893, 2.26893),
    "L_elbow": (0.0, 2.53073),
    "R_shoulder_pitch": (-1.5708, 3.14159),
    "R_shoulder_roll": (0.0, 3.14159),
    "R_shoulder_yaw": (-2.26893, 2.26893),
    "R_elbow": (0.0, 1.5708),
    "L_hip_yaw": (-0.785398, 0.785398),
    "L_hip_roll": (-0.261799, 0.785398),
    "L_hip_pitch": (-0.523599, 2.00713),
    "L_knee_pitch": (-2.44346, 0.0872665),
    "L_ankle_pitch": (-1.39626, 1.0472),
    "R_hip_yaw": (-0.785398, 0.785398),
    "R_hip_roll": (-0.261799, 0.785398),
    "R_hip_pitch": (-0.523599, 2.00713),
    "R_knee_pitch": (-2.44346, 0.0872665),
    "R_ankle_pitch": (-1.39626, 1.0472),
}

X02LITE_DOF_NAMES = list(X02LITE_HEALTH_LIMITS.keys())


def _expected_csv_columns_for_robot(robot: str | None) -> int | None:
    key = (robot or "").strip().lower()
    if key == "g1":
        key = "unitree_g1"
    elif key == "h1":
        key = "unitree_h1"
    elif key in ("x02", "x02lite"):
        key = "x02lite"
    elif key in ("openloong", "loong"):
        key = "openloong"
    return EXPECTED_CSV_COLUMNS_BY_ROBOT.get(key)


def _build_unity_csv_row(qpos, robot: str, csv_path: str, frame_id: int):
    q = np.asarray(qpos)
    row = np.concatenate([q[:3], q[3:7][[1, 2, 3, 0]], q[7:]], axis=0)
    expected_cols = _expected_csv_columns_for_robot(robot)
    if expected_cols is not None and int(row.shape[0]) != expected_cols:
        raise RuntimeError(
            f"[GMR] CSV column mismatch: robot={robot} expected_cols={expected_cols} "
            f"actual_cols={int(row.shape[0])} qpos_len={len(q)} frame={frame_id} csv={csv_path}. "
            "Check ROBOT env/--robot and the loaded MuJoCo XML before feeding Unity."
        )
    return row


def _write_unity_qpos_metadata(state):
    """Write a sidecar describing the exact MuJoCo qpos contract Unity consumes.

    This intentionally does not change live_motion.csv. Unity can use the JSON
    to validate joint names/order, qpos0/home offsets, limits, and root quat
    convention instead of guessing from scene traversal order.
    """
    if state is None or state.get("metadata_written"):
        return

    csv_path = state.get("csv_path")
    retarget = state.get("retarget")
    if not csv_path or retarget is None or getattr(retarget, "model", None) is None:
        return

    model = retarget.model
    joints = []
    for joint_id in range(int(model.njnt)):
        qpos_adr = int(model.jnt_qposadr[joint_id])
        if qpos_adr < 7:
            continue

        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id) or f"qpos_{qpos_adr}"
        limited = bool(model.jnt_limited[joint_id]) if hasattr(model, "jnt_limited") else False
        joint_range = None
        if limited:
            joint_range = [
                float(model.jnt_range[joint_id][0]),
                float(model.jnt_range[joint_id][1]),
            ]

        joints.append({
            "csvIndex": qpos_adr - 7,
            "jointName": joint_name,
            "qposAdr": qpos_adr,
            "qpos0": float(model.qpos0[qpos_adr]),
            "limited": limited,
            "range": joint_range,
        })

    joints.sort(key=lambda item: item["csvIndex"])
    metadata = {
        "robot": state.get("robot"),
        "csvColumns": int(model.nq),
        "qposLength": int(model.nq),
        "root": {
            "position": "qpos[0:3]",
            "quatSourceOrder": "wxyz",
            "csvQuatOrder": "xyzw",
            "unityMapping": "pos=(-y,z,x), quat=(-y,z,x,-w)",
            "qpos0": [float(v) for v in np.asarray(model.qpos0[:7]).tolist()],
        },
        "joints": joints,
    }

    metadata_dir = os.path.dirname(csv_path) or "."
    metadata_path = os.path.join(metadata_dir, "retarget_metadata.json")
    os.makedirs(metadata_dir, exist_ok=True)
    tmp_path = metadata_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    os.replace(tmp_path, metadata_path)

    state["metadata_path"] = metadata_path
    state["metadata_written"] = True
    logger.info(
        f"[GMR] wrote Unity qpos metadata: robot={state.get('robot')} "
        f"csv_cols={metadata['csvColumns']} joints={len(joints)} path={metadata_path}"
    )


def _x02lite_fk_ankles(qpos, postprocessor, state):
    if postprocessor is None or getattr(postprocessor, "kinematics_model", None) is None:
        return None, None

    kinematics_model = postprocessor.kinematics_model
    cached = state.get("_x02lite_ankle_indices")
    if cached is None:
        body_names = list(getattr(kinematics_model, "body_names", []))
        if "L_ankle_pitch_Link" not in body_names or "R_ankle_pitch_Link" not in body_names:
            return None, None
        cached = (
            body_names.index("L_ankle_pitch_Link"),
            body_names.index("R_ankle_pitch_Link"),
        )
        state["_x02lite_ankle_indices"] = cached

    idx_l, idx_r = cached
    device = postprocessor.device
    q = np.asarray(qpos, dtype=np.float32)
    root_pos = torch.from_numpy(q[:3][None]).to(device=device, dtype=torch.float32)
    root_rot_xyzw = torch.from_numpy(q[3:7][[1, 2, 3, 0]][None]).to(device=device, dtype=torch.float32)
    dof_pos = torch.from_numpy(q[7:][None]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        body_pos, _ = kinematics_model.forward_kinematics(root_pos, root_rot_xyzw, dof_pos)
    left = body_pos[0, idx_l].detach().cpu().numpy()
    right = body_pos[0, idx_r].detach().cpu().numpy()
    return left, right


def _log_x02lite_qpos_health(qpos, postprocessor, frame_id, state):
    q = np.asarray(qpos, dtype=np.float64)
    dof = q[7:]
    if dof.shape[0] != len(X02LITE_DOF_NAMES):
        logger.warning(
            f"[X02Lite/Health] unexpected dof count={dof.shape[0]} "
            f"(expected {len(X02LITE_DOF_NAMES)}) frame={frame_id}"
        )
        return

    margin = np.deg2rad(5.0)
    saturated = []
    for idx, name in enumerate(X02LITE_DOF_NAMES):
        low, high = X02LITE_HEALTH_LIMITS[name]
        value = float(dof[idx])
        if value <= low + margin or value >= high - margin:
            saturated.append(f"{name}={value:.3f}")

    root_z = float(q[2])
    health_parts = [f"root_z={root_z:.3f}"]

    if root_z < 0.82 or root_z > 1.08:
        health_parts.append("root_height_out_of_band")

    try:
        left_ankle, right_ankle = _x02lite_fk_ankles(q, postprocessor, state)
        if left_ankle is not None and right_ankle is not None:
            lateral_sep = abs(float(left_ankle[1] - right_ankle[1]))
            planar_sep = float(np.linalg.norm(left_ankle[:2] - right_ankle[:2]))
            health_parts.append(f"ankle_lateral_sep={lateral_sep:.3f}")
            health_parts.append(f"ankle_planar_sep={planar_sep:.3f}")
            if lateral_sep < 0.05 or planar_sep < 0.08:
                health_parts.append("ankles_too_close")
    except Exception as exc:
        health_parts.append(f"ankle_fk_error={exc}")

    if saturated:
        health_parts.append("near_limits=" + ";".join(saturated[:8]))

    logger.info(f"[X02Lite/Health] frame={frame_id} " + " ".join(health_parts))

def _align_tail_betas(raw_betas, body_model):
    betas = np.asarray(raw_betas, dtype=np.float32).reshape(-1)
    target_dim = int(getattr(body_model, "num_betas", betas.shape[0]))
    if betas.shape[0] < target_dim:
        betas = np.pad(betas, (0, target_dim - betas.shape[0]))
    elif betas.shape[0] > target_dim:
        betas = betas[:target_dim]
    return betas

def _tail_apply_yup_to_zup(root_orient, trans):
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    rot_fix = R.from_matrix(rotation_matrix)
    root_orient = (rot_fix * R.from_rotvec(root_orient)).as_rotvec().astype(np.float32)
    trans = trans @ rotation_matrix.T
    return root_orient, trans


class TcpStreamSender:
    """Send JPEG frames to Unity StreamReceiver via TCP (127.0.0.1:9876)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9876):
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def connect(self) -> bool:
        try:
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock.settimeout(3.0)
            self._sock.connect((self._host, self._port))
            self._sock.settimeout(5.0)
            logger.info(f"[TCP] Connected to Unity at {self._host}:{self._port}")
            return True
        except Exception as e:
            logger.warning(f"[TCP] Cannot connect to Unity ({e}), will retry")
            self._sock = None
            return False

    def send_frame(self, stream_id: int, bgr: "np.ndarray", quality: int = 80) -> bool:
        with self._lock:
            if self._sock is None:
                if not self.connect():
                    return False
            try:
                ok, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    return False
                data = jpg.tobytes()
                header = struct.pack('>II', stream_id, len(data))
                self._sock.sendall(header + data)
                return True
            except (socket.error, BrokenPipeError, ConnectionResetError, OSError):
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
                return False

    def close(self):
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None

    @property
    def connected(self) -> bool:
        return self._sock is not None


import sys
import select
import signal
import torch
import numpy as np
from loguru import logger
from queue import Queue, Full, Empty
from contextlib import nullcontext

try:
    import termios
    import tty
except Exception:
    termios = None
    tty = None

import mujoco as mj
from configs.config import get_cfg_defaults
try:
    from lib.vis.renderer import Renderer
    RENDERER_IMPORT_ERROR = None
except Exception as e:
    Renderer = None
    RENDERER_IMPORT_ERROR = e
import imageio
from collections import deque
from lib.data.utils.normalizer import Normalizer
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.utils.kp_utils import root_centering
from lib.models.preproc.backbone.utils import process_image
from scripts.stream_demo_common import (
    TailStreamWriter,
    env_flag,
    estimate_fps_from_timestamps,
    init_preview_window,
    rotate_frame_bgr,
    sanitize_preview_frame,
)

class _HeadlessViewer:
    """Minimal MuJoCo viewer for headless offscreen rendering (no GLFW/X11 needed).

    Replicates the essential API of RobotMotionViewer without any display
    dependencies.  Used for screenshot capture and TCP streaming on headless servers.
    """

    def __init__(self, robot_type, camera_follow=True, motion_fps=30,
                 camera_lookat_height_offset=0.50, camera_elevation=-28.0,
                 camera_distance_scale=1.70, camera_azimuth=None,
                 robot_path=None, **_kwargs):
        from general_motion_retargeting.params import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT

        xml_path = robot_path if robot_path and str(robot_path).strip() else ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)

        self.motion_fps = motion_fps
        self.camera_follow = camera_follow
        self.camera_lookat_height_offset = float(camera_lookat_height_offset)
        self.camera_elevation = -abs(float(camera_elevation))
        self.camera_distance_scale = float(camera_distance_scale)
        self.camera_azimuth = camera_azimuth

        self._cam = mj.MjvCamera()
        self._cam.lookat[:] = [0.0, 0.0, 1.0]
        self._cam.distance = self.viewer_cam_distance * max(0.1, self.camera_distance_scale)
        self._cam.elevation = self.camera_elevation
        self._cam.azimuth = float(self.camera_azimuth) if self.camera_azimuth is not None else 90.0

        self.viewer = type('_CamHolder', (), {'cam': self._cam})()

    def _camera_lookat_from_robot_bounds(self):
        positions = np.asarray(self.data.xpos[1:], dtype=np.float64)
        if positions.size == 0:
            return np.asarray(self.data.qpos[:3], dtype=np.float64).copy(), 0.5

        finite_mask = np.isfinite(positions).all(axis=1)
        positions = positions[finite_mask]
        if positions.size == 0:
            return np.asarray(self.data.qpos[:3], dtype=np.float64).copy(), 0.5

        bounds_min = positions.min(axis=0)
        bounds_max = positions.max(axis=0)
        lookat = (bounds_min + bounds_max) * 0.5
        radius = float(np.linalg.norm(bounds_max - bounds_min) * 0.5)
        lookat[2] += self.camera_lookat_height_offset
        return lookat, max(radius, 0.5)

    def _update_follow_camera(self):
        lookat, radius = self._camera_lookat_from_robot_bounds()
        self._cam.lookat[:] = lookat
        base_distance = self.viewer_cam_distance * max(0.1, self.camera_distance_scale)
        self._cam.distance = max(base_distance, radius * 2.7)
        self._cam.elevation = self.camera_elevation
        if self.camera_azimuth is not None:
            self._cam.azimuth = float(self.camera_azimuth)

    def step(self, root_pos, root_rot, dof_pos, human_motion_data=None,
             show_human_body_name=False, human_pos_offset=None, rate_limit=False,
             follow_camera=True, human_point_scale=0.1):
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot
        self.data.qpos[7:] = dof_pos
        mj.mj_forward(self.model, self.data)

        if follow_camera:
            self._update_follow_camera()

    def close(self):
        pass


STREAM_SEQ_LEN = 16

def xyxy_to_cxcys(bbox, s_factor=1.2):
    cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200 * s_factor
    return np.array([cx, cy, scale])


def run_stream_mt(
    cfg,
    video_path,
    output_dir,
    rotate_deg=None,
    record_whamvideo=False,
    capture_time=0.0,
    gmr_args=None,
    stream_npz_dir=None,
    capture_interval=0,
    capture_dir="results/screenshots",
):
    os.makedirs(output_dir, exist_ok=True)

    is_webcam = video_path == '0' or video_path == 0

    # Screenshot capture setup
    cap_orig = None
    if capture_interval > 0 and not is_webcam:
        os.makedirs(capture_dir, exist_ok=True)
        cap_orig = cv2.VideoCapture(video_path)
        if not cap_orig.isOpened():
            logger.warning(f"Cannot open {video_path} for original-frame capture; screenshots disabled.")
            cap_orig = None
            capture_interval = 0
        else:
            # Create subdirectories for each screenshot layer
            for sub in ("orig", "wham", "mujoco"):
                os.makedirs(os.path.join(capture_dir, sub), exist_ok=True)
            logger.info(f"Capture screenshots every {capture_interval} frames → {capture_dir}")

    device = cfg.DEVICE.lower()
    is_cuda_device = str(device).startswith('cuda') and torch.cuda.is_available()
    use_amp = env_flag('WHAM_USE_AMP', True) and is_cuda_device
    detect_interval = max(1, int(os.environ.get('WHAM_DETECT_INTERVAL', '2')))
    infer_interval = max(1, int(os.environ.get('WHAM_INFER_INTERVAL', '1')))
    stream_seq_len = max(4, int(os.environ.get('WHAM_STREAM_SEQ_LEN', str(STREAM_SEQ_LEN))))
    tcp_stream_wham = env_flag('TCP_STREAM_WHAM', True)
    tcp_stream_gmr = env_flag('TCP_STREAM_GMR', True)

    if is_cuda_device:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.allow_tf32 = True

    def autocast_ctx():
        if use_amp:
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return nullcontext()

    root_dir = os.path.abspath(os.path.dirname(__file__))
    webcam_capture_time = max(0.0, float(capture_time))
    webcam_start_ts = None
    preview_window_name = 'WHAM Stream'
    preview_window_enabled = bool(record_whamvideo)
    preview_max_w = max(320, int(os.environ.get('WHAM_PREVIEW_MAX_WIDTH', '960')))
    preview_max_h = max(240, int(os.environ.get('WHAM_PREVIEW_MAX_HEIGHT', '540')))
    logger.info(
        f"Perf opts: amp={int(use_amp)} detect_interval={detect_interval} "
        f"infer_interval={infer_interval} seq_len={stream_seq_len} device={device} "
        "stream_mode=tail"
    )
    heartbeat_frames = max(1, int(os.environ.get('PIPELINE_HEARTBEAT_FRAMES', '30')))
    low_memory_mode = env_flag("WHAM_LOW_MEMORY_MODE", False)
    pipeline_queue_size = max(2, int(os.environ.get("WHAM_PIPELINE_QUEUE_SIZE", "10")))
    gmr_preview_fps = max(0.5, float(os.environ.get("GMR_PREVIEW_FPS", "3.0")))
    gmr_preview_interval = 1.0 / gmr_preview_fps
    logger.info(
        f"[Input] source={'webcam:0' if is_webcam else video_path} capture_time={webcam_capture_time:.2f}s "
        f"record_whamvideo={int(record_whamvideo)} tcp_requested={int(bool(getattr(gmr_args, 'tcp', False)))} "
        f"tcp_stream_wham={int(tcp_stream_wham)} tcp_stream_gmr={int(tcp_stream_gmr)} "
        f"heartbeat_frames={heartbeat_frames} gmr_preview_fps={gmr_preview_fps:.2f} "
        f"low_memory={int(low_memory_mode)} queue_size={pipeline_queue_size}"
    )
    if preview_window_enabled and os.name != 'nt' and not os.environ.get('DISPLAY'):
        logger.warning("RECORD_WHAMVIDEO=1 but DISPLAY is not set; preview window disabled.")
        preview_window_enabled = False
    
    # Init Models
    logger.info("Loading models...")
    pose_model_cfg = os.path.join(
        root_dir,
        'third-party',
        'ViTPose',
        'configs',
        'body',
        '2d_kpt_sview_rgb_img',
        'topdown_heatmap',
        'coco',
        'ViTPose_base_coco_256x192.py'
    )
    pose_model_ckpt = os.path.join(root_dir, 'checkpoints', 'vitpose_base_coco_aic_mpii.pth')
    bbox_model_ckpt = os.path.join(root_dir, 'checkpoints', 'yolov8n.pt')
    detector = DetectionModel(
        device,
        pose_model_cfg=pose_model_cfg,
        pose_model_ckpt=pose_model_ckpt,
        bbox_model_ckpt=bbox_model_ckpt,
    )
    extractor = FeatureExtractor(device, cfg.FLIP_EVAL)
    smpl = build_body_model('cpu')
    network = build_network(cfg, smpl)
    network.eval()

    if use_amp:
        network = network.half()
        extractor.model = extractor.model.half()
        smpl = smpl.float()  # shared ref was converted by network.half(); revert for build_wham_inits
        logger.info("FP16: WHAM network + HMR2 extractor converted to half precision")

    keypoints_normalizer = Normalizer(cfg)

    if is_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Faild to load video file {video_path}'
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if rotate_deg is None and (not is_webcam) and hasattr(cv2, 'CAP_PROP_ORIENTATION_AUTO'):
        orientation_meta = None
        if hasattr(cv2, 'CAP_PROP_ORIENTATION_META'):
            orientation_meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        if orientation_meta is not None:
            logger.info(f'Input video orientation metadata: {orientation_meta:.0f} deg (auto-rotation enabled)')

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    first_frame = None
    ok, probe = cap.read()
    assert ok and probe is not None, f'Failed to read first frame from {video_path}'
    if rotate_deg is not None:
        probe = rotate_frame_bgr(probe, rotate_deg)

    input_scale = float(os.environ.get('WHAM_INPUT_SCALE', '1.0'))
    input_scale = float(np.clip(input_scale, 0.1, 1.0))
    input_scale = float(min(input_scale, 1.0))
    apply_input_resize = input_scale < 0.999
    if apply_input_resize:
        new_w = max(2, int(round(probe.shape[1] * input_scale)))
        new_h = max(2, int(round(probe.shape[0] * input_scale)))
        probe = cv2.resize(probe, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Input resized for speed: scale={input_scale:.3f}, size={new_w}x{new_h}")

    first_frame = probe
    if is_webcam:
        webcam_start_ts = time.time()
    height, width = first_frame.shape[:2]
    logger.info(f"[Input] first frame read; fps={fps:.2f}, effective_size={width}x{height} (scale={input_scale:.3f})")

    def gmr_renderer_size(for_capture: bool = False):
        if for_capture and not bool(getattr(gmr_args, "tcp", False)):
            return 480, 640, "screenshot capture"

        base_h = max(120, int(os.environ.get("GMR_TCP_RENDER_HEIGHT", "720")))
        requested_w = int(os.environ.get("GMR_TCP_RENDER_WIDTH", "0") or "0")
        aspect = max(0.1, min(10.0, width / max(float(height), 1.0)))
        base_w = max(2, requested_w if requested_w > 0 else int(round(base_h * aspect)))
        return base_h, base_w, f"Unity streaming aspect={aspect:.3f}"

    def gmr_window_size():
        max_w = max(320, int(os.environ.get("GMR_VIEWER_MAX_WIDTH", str(preview_max_w))))
        max_h = max(240, int(os.environ.get("GMR_VIEWER_MAX_HEIGHT", str(preview_max_h))))
        scale = min(max_w / max(float(width), 1.0), max_h / max(float(height), 1.0), 1.0)
        return max(1, int(round(width * scale))), max(1, int(round(height * scale)))

    if preview_window_enabled:
        logger.info("WHAM preview window will be initialized in render thread.")
    
    from lib.utils.imutils import compute_cam_intrinsics
    res = torch.tensor([width, height]).float()
    intrinsics = compute_cam_intrinsics(res)
    intrinsics_batch = intrinsics.unsqueeze(0).to(device)
    res_batch = res.unsqueeze(0).to(device)
    import lib.utils.transforms as pt_transforms

    renderer = None
    wham_video_path = os.path.join(output_dir, 'output.mp4')
    if record_whamvideo:
        if Renderer is None:
            logger.warning(f"pytorch3d renderer unavailable ({RENDERER_IMPORT_ERROR}); continue without mesh overlay.")
        else:
            focal_length = (width ** 2 + height ** 2) ** 0.5
            try:
                renderer = Renderer(width, height, focal_length, device, smpl.faces)
            except Exception as e:
                logger.warning(f"Failed to initialize renderer ({e}); continue without mesh overlay.")
                renderer = None
        if is_webcam:
            logger.info("Webcam WHAM writer will use measured runtime FPS for output video.")

    tail_writer = None
    if stream_npz_dir is not None:
        tail_writer = TailStreamWriter(stream_npz_dir, flush_interval=1)
        logger.info(f"Tail stream enabled: {tail_writer.tail_path} (flush_interval={tail_writer.flush_interval})")


    # Queues for pipeline
    read_Q = Queue(maxsize=pipeline_queue_size)
    det_Q = Queue(maxsize=pipeline_queue_size)
    ext_Q = Queue(maxsize=pipeline_queue_size)
    wham_Q = Queue(maxsize=pipeline_queue_size)
    render_Q = Queue(maxsize=max(2, int(os.environ.get('WHAM_RENDER_QUEUE_SIZE', '16'))))
    render_thread = None
    render_drop_count = 0

    def safe_qsize(q):
        try:
            return q.qsize()
        except Exception:
            return -1

    thread_failure_lock = threading.Lock()
    thread_failure = {"message": None}

    def guarded_thread(name, target):
        try:
            target()
        except Exception as exc:
            with thread_failure_lock:
                if thread_failure["message"] is None:
                    thread_failure["message"] = f"{name} thread failed: {exc}"
            logger.exception(f"[FATAL] {name} thread failed")
            request_stop()

    def extract_latest_vertices(pred):
        verts = pred['verts_cam']
        trans = pred['trans_cam']

        if verts.ndim == 4:
            verts = verts[0, -1]
        elif verts.ndim == 3:
            verts = verts[-1]
        elif verts.ndim != 2:
            raise ValueError(f"Unexpected verts_cam shape: {tuple(verts.shape)}")

        if trans.ndim == 3:
            trans = trans[0, -1]
        elif trans.ndim == 2:
            trans = trans[-1]
        elif trans.ndim != 1:
            raise ValueError(f"Unexpected trans_cam shape: {tuple(trans.shape)}")

        trans = trans.reshape(-1)
        if trans.numel() != 3:
            raise ValueError(f"Unexpected trans_cam flattened size: {trans.numel()}")

        return verts + trans.unsqueeze(0)

    def extract_latest_body_pose(pose):
        if pose.ndim == 5:
            pose = pose[0, -1]
        elif pose.ndim == 4:
            pose = pose[-1]
        elif pose.ndim != 3:
            raise ValueError(f"Unexpected poses_body shape: {tuple(pose.shape)}")
        return pose

    def extract_latest_root_pose(root):
        if root.ndim == 4:
            root = root[0, -1]
        elif root.ndim == 3:
            root = root[-1]
        elif root.ndim != 2:
            raise ValueError(f"Unexpected root pose shape: {tuple(root.shape)}")
        return root

    def extract_latest_vec(vec, name):
        if vec.ndim == 3:
            vec = vec[0, -1]
        elif vec.ndim == 2:
            vec = vec[-1]
        elif vec.ndim != 1:
            raise ValueError(f"Unexpected {name} shape: {tuple(vec.shape)}")
        return vec.reshape(-1)

    def to_body_pose63(pose_aa):
        pose_aa = pose_aa.reshape(-1)
        if pose_aa.numel() >= 63:
            return pose_aa[:63]
        padded = torch.zeros(63, dtype=pose_aa.dtype, device=pose_aa.device)
        padded[:pose_aa.numel()] = pose_aa
        return padded

    def to_betas10(betas):
        if betas.numel() >= 10:
            return betas[:10]
        padded = torch.zeros(10, dtype=betas.dtype, device=betas.device)
        padded[:betas.numel()] = betas
        return padded

    def extract_latest_smplx_params(pred):
        body_pose_mat = extract_latest_body_pose(pred['poses_body'])
        root_cam_mat = extract_latest_root_pose(pred['poses_root_cam'])
        trans_cam = extract_latest_vec(pred['trans_cam'], 'trans_cam')
        betas = to_betas10(extract_latest_vec(pred['betas'], 'betas'))

        body_pose_aa = to_body_pose63(matrix_to_axis_angle(body_pose_mat).reshape(-1))
        _ = matrix_to_axis_angle(root_cam_mat).reshape(-1)

        if 'poses_root_world' in pred:
            root_world_aa = matrix_to_axis_angle(extract_latest_root_pose(pred['poses_root_world'])).reshape(-1)
        else:
            root_world_aa = matrix_to_axis_angle(root_cam_mat).reshape(-1)

        if 'trans_world' in pred:
            trans_world = extract_latest_vec(pred['trans_world'], 'trans_world')
        else:
            trans_world = trans_cam.clone()

        result = {
            'body_pose': body_pose_aa.detach().cpu().numpy(),
            'betas': betas.detach().cpu().numpy(),
            'global_orient_global': root_world_aa.detach().cpu().numpy(),
            'transl_global': trans_world.detach().cpu().numpy(),
        }

        # Extract root velocity for continuous pelvis tracking across WHAM windows.
        # WHAM resets trans_world (cumsum from zero) each sliding window because
        # we call with states=None.  Instead, integrate pred_vel ourselves.
        if 'vel_root' in pred:
            vel_root = extract_latest_vec(pred['vel_root'], 'vel_root')
            result['vel_root'] = vel_root.detach().cpu().numpy()
        if 'poses_root_world' in pred:
            result['poses_root_world_mat'] = extract_latest_root_pose(
                pred['poses_root_world']
            ).detach().cpu().numpy()

        return result

    def request_stop():
        stop_event.set()
        try:
            cap.release()
        except Exception:
            pass
        # Wake up any thread blocked on queue.get(). If a queue is full,
        # drop one item to make room for the sentinel so stop can propagate.
        for q in (read_Q, det_Q, ext_Q, wham_Q, render_Q):
            injected = False
            for _ in range(3):
                try:
                    q.put_nowait(None)
                    injected = True
                    break
                except Full:
                    try:
                        q.get_nowait()
                    except Empty:
                        break
            if not injected:
                try:
                    q.put(None, timeout=0.05)
                except Exception:
                    pass
    
    stop_event = threading.Event()
    terminal_stop_event = threading.Event()

    def terminal_key_listener():
        if not is_webcam or webcam_capture_time > 0:
            return
        if (not sys.stdin.isatty()) or termios is None or tty is None:
            logger.warning("Terminal key listener unavailable (non-interactive stdin). Use Ctrl+C to stop webcam stream.")
            return

        logger.info("Webcam mode started. Press 'q' or ESC in terminal to stop recording.")
        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            typed_buffer = ""
            while not stop_event.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                ch = sys.stdin.read(1)
                if ch in ('q', 'Q', '\x1b'):
                    logger.info("User requested exit from terminal input.")
                    terminal_stop_event.set()
                    request_stop()
                    break
                typed_buffer = (typed_buffer + ch.lower())[-3:]
                if typed_buffer == 'esc':
                    logger.info("User requested exit from terminal input.")
                    terminal_stop_event.set()
                    request_stop()
                    break
        except Exception as e:
            logger.warning(f"Terminal key listener stopped: {e}")
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
            except Exception:
                pass

    def render_thread_worker():
        nonlocal render_drop_count
        writer_local = None
        writer_buffer = []
        writer_buffer_ts = []
        local_preview_enabled = bool(preview_window_enabled)

        if local_preview_enabled:
            try:
                cv2.startWindowThread()
            except Exception:
                pass
            try:
                init_w, init_h = init_preview_window(
                    preview_window_name,
                    width,
                    height,
                    preview_max_w,
                    preview_max_h,
                )
                logger.info(f"WHAM preview window initialized at {init_w}x{init_h}.")
            except Exception as e:
                logger.warning(f"Failed to initialize preview window ({e}); preview disabled.")
                local_preview_enabled = False

        def ensure_webcam_writer_local(force=False):
            nonlocal writer_local, writer_buffer, writer_buffer_ts
            if (not record_whamvideo) or (not is_webcam):
                return
            if writer_local is not None:
                return
            if len(writer_buffer) == 0:
                return
            if (not force) and len(writer_buffer) < 15:
                return

            est_fps = estimate_fps_from_timestamps(writer_buffer_ts, fallback_fps=max(1.0, float(fps)))
            writer_local = imageio.get_writer(
                wham_video_path,
                fps=float(est_fps),
                mode='I',
                format='FFMPEG',
                macro_block_size=1,
            )
            for f in writer_buffer:
                writer_local.append_data(f)
            logger.info(f"Initialized webcam WHAM writer at {est_fps:.2f} FPS.")
            writer_buffer = []
            writer_buffer_ts = []

        if record_whamvideo and (not is_webcam):
            writer_local = imageio.get_writer(
                wham_video_path,
                fps=float(fps),
                mode='I',
                format='FFMPEG',
                macro_block_size=1,
            )

        while not stop_event.is_set():
            try:
                data = render_Q.get(timeout=0.1)
            except Empty:
                continue
            if data is None:
                break

            frame, frame_ts = data
            frame = sanitize_preview_frame(frame)
            frame_rgb_out = np.ascontiguousarray(frame[..., ::-1])

            if is_webcam:
                if writer_local is None:
                    writer_buffer.append(frame_rgb_out.copy())
                    writer_buffer_ts.append(float(frame_ts))
                    ensure_webcam_writer_local(force=False)
                else:
                    writer_local.append_data(frame_rgb_out)
            elif writer_local is not None:
                writer_local.append_data(frame_rgb_out)

            if local_preview_enabled:
                try:
                    cv2.imshow(preview_window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                except Exception as e:
                    logger.warning(f"OpenCV preview failed: {e}")
                    key = -1

                if key == 27 or key == ord('q'):
                    logger.info("User requested exit from preview window.")
                    terminal_stop_event.set()
                    request_stop()
                    break

        if record_whamvideo and is_webcam:
            ensure_webcam_writer_local(force=True)
        if writer_local is not None:
            writer_local.close()

        if local_preview_enabled:
            try:
                cv2.destroyWindow(preview_window_name)
            except Exception:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
    
    def reader_thread():
        frame_id = 0
        if first_frame is not None:
            while not stop_event.is_set():
                try:
                    read_Q.put((frame_id, time.time(), first_frame), timeout=0.1)
                    break
                except Full:
                    continue
            logger.info(f"[Input] queued first frame={frame_id} read_Q={safe_qsize(read_Q)}")
            frame_id += 1

        while cap.isOpened() and not stop_event.is_set():
            if is_webcam and webcam_capture_time > 0 and webcam_start_ts is not None:
                if (time.time() - webcam_start_ts) >= webcam_capture_time:
                    logger.info(f"Reached webcam capture limit ({webcam_capture_time:.2f}s), stopping.")
                    break
            ret, frame = cap.read()
            if not ret: break
            if rotate_deg is not None:
                frame = rotate_frame_bgr(frame, rotate_deg)
            if apply_input_resize:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            while not stop_event.is_set():
                try:
                    read_Q.put((frame_id, time.time(), frame), timeout=0.1)
                    break
                except Full:
                    continue
            if frame_id % heartbeat_frames == 0:
                logger.info(f"[Input] queued frame={frame_id} read_Q={safe_qsize(read_Q)}")
            frame_id += 1
        try:
            read_Q.put(None, timeout=0.2)
        except Exception:
            pass
    
    def detector_thread():
        detector.initialize_tracking()
        selected_track_id = None
        last_detect_frame = -10**9
        last_track_id = None
        last_kp2d = None
        last_bbox_c = None

        while not stop_event.is_set():
            try:
                data = read_Q.get(timeout=0.1)
            except Empty:
                continue
            if data is None:
                try:
                    det_Q.put(None, timeout=0.2)
                except Exception:
                    pass
                break
            frame_id, start_time, frame = data

            track_id = None
            kp2d = None
            bbox_c = None
            poses_count = -1
            valid_count = -1

            need_detect = (frame_id - last_detect_frame) >= detect_interval or last_track_id is None
            if need_detect:
                detector.track(frame, fps, 1000)
                poses = detector.pose_results_last
                poses_count = len(poses)
                valid_count = 0

                if len(poses) > 0:
                    valid_poses = []
                    for pose in poses:
                        valid = (pose['keypoints'][:, -1] > 0.3).sum()
                        if valid >= 6:
                            valid_poses.append(pose)
                    valid_count = len(valid_poses)

                    best_pose = None
                    if len(valid_poses) > 0:
                        if selected_track_id is not None:
                            for pose in valid_poses:
                                if pose['track_id'] == selected_track_id:
                                    best_pose = pose
                                    break

                        if best_pose is None:
                            best_pose = max(
                                valid_poses,
                                key=lambda p: (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1])
                            )

                    if best_pose is not None:
                        selected_track_id = best_pose['track_id']
                        track_id = best_pose['track_id']
                        bbox_c = xyxy_to_cxcys(best_pose['bbox'])
                        kp2d = best_pose['keypoints']

                last_detect_frame = frame_id
                if track_id is not None and kp2d is not None and bbox_c is not None:
                    last_track_id = int(track_id)
                    last_kp2d = kp2d.copy()
                    last_bbox_c = bbox_c.copy()
                else:
                    last_track_id = None
                    last_kp2d = None
                    last_bbox_c = None
            else:
                track_id = last_track_id
                kp2d = None if last_kp2d is None else last_kp2d.copy()
                bbox_c = None if last_bbox_c is None else last_bbox_c.copy()

            if frame_id == 0 or frame_id % heartbeat_frames == 0:
                logger.info(
                    f"[Detect] frame={frame_id} need_detect={int(need_detect)} poses={poses_count} "
                    f"valid={valid_count} selected_track={selected_track_id} track_id={track_id} det_Q={safe_qsize(det_Q)}"
                )
            
            while not stop_event.is_set():
                try:
                    det_Q.put((frame_id, start_time, frame, track_id, kp2d, bbox_c), timeout=0.1)
                    break
                except Full:
                    continue
            
    def extractor_thread():
        init_cache = {}
        while not stop_event.is_set():
            try:
                data = det_Q.get(timeout=0.1)
            except Empty:
                continue
            if data is None:
                try:
                    ext_Q.put(None, timeout=0.2)
                except Exception:
                    pass
                break
            frame_id, start_time, frame, track_id, kp2d, bbox_c = data
            
            if track_id is not None:
                cx, cy, scale = bbox_c
                norm_img, _ = process_image(frame[..., ::-1], [cx, cy], scale, 256, 256)
                norm_img = torch.from_numpy(norm_img).unsqueeze(0).to(device, non_blocking=True)

                with torch.inference_mode():
                    with autocast_ctx():
                        feature = extractor.model(norm_img, encode=True)

                init_data = init_cache.get(track_id)
                if init_data is None:
                    dummy_tracking = {track_id: {}}
                    with torch.inference_mode():
                        with autocast_ctx():
                            extractor.predict_init(norm_img, dummy_tracking, track_id, flip_eval=False)
                    init_data = dummy_tracking[track_id]
                    init_cache[track_id] = init_data
                
                while not stop_event.is_set():
                    try:
                        ext_Q.put((frame_id, start_time, frame, track_id, kp2d, bbox_c, feature, init_data), timeout=0.1)
                        break
                    except Full:
                        continue
                if frame_id == 0 or frame_id % heartbeat_frames == 0:
                    logger.info(f"[Extract] frame={frame_id} track_id={track_id} feature=1 ext_Q={safe_qsize(ext_Q)}")
            else:
                while not stop_event.is_set():
                    try:
                        ext_Q.put((frame_id, start_time, frame, None, None, None, None, None), timeout=0.1)
                        break
                    except Full:
                        continue
                if frame_id == 0 or frame_id % heartbeat_frames == 0:
                    logger.warning(f"[Extract] frame={frame_id} has no tracked person; ext_Q={safe_qsize(ext_Q)}")
                
    def build_wham_inits(init_data_hist, first_norm_kp2d):
        with torch.inference_mode():
            init_output = smpl.get_output(
                global_orient=init_data_hist['init_global_orient'].float().to(device),
                body_pose=init_data_hist['init_body_pose'].float().to(device),
                betas=init_data_hist['init_betas'].float().to(device),
                pose2rot=False,
                return_full_pose=True
            )

        init_kp3d = root_centering(init_output.joints[:, :17], 'coco')
        init_kp = torch.cat(
            (init_kp3d.reshape(1, -1), first_norm_kp2d.reshape(1, -1)),
            dim=-1
        ).unsqueeze(0).to(device)
        init_smpl = pt_transforms.matrix_to_rotation_6d(init_output.full_pose).unsqueeze(0).to(device)
        init_root = pt_transforms.matrix_to_rotation_6d(init_output.global_orient).to(device)
        return (init_kp, init_smpl), init_root

    def wham_thread():
        temporal_history = {}

        while not stop_event.is_set():
            try:
                data = ext_Q.get(timeout=0.1)
            except Empty:
                continue
            if data is None:
                try:
                    wham_Q.put(None, timeout=0.2)
                except Exception:
                    pass
                break
            frame_id, start_time, frame, track_id, kp2d, bbox_c, feature, init_data = data
            
            if track_id is not None:
                if track_id not in temporal_history:
                    temporal_history[track_id] = {
                        'kp2d': deque(maxlen=stream_seq_len),
                        'bbox': deque(maxlen=stream_seq_len),
                        'feature': deque(maxlen=stream_seq_len),
                        'init_data': init_data,
                        'cached_inits': None,
                        'cached_init_root': None,
                        'last_pred': None,
                    }

                hist = temporal_history[track_id]
                hist['kp2d'].append(kp2d)
                hist['bbox'].append(bbox_c)
                hist['feature'].append(feature.squeeze(0).detach())

                kp2d_t = torch.from_numpy(np.stack(list(hist['kp2d']), axis=0)).float()
                mask = (kp2d_t[..., -1] < 0.3).unsqueeze(0).to(device)
                bbox_t = torch.from_numpy(np.stack(list(hist['bbox']), axis=0)).float()
                
                norm_kp2d, _ = keypoints_normalizer(
                    kp2d_t[..., :-1].clone(), res, intrinsics, 224, 224, bbox_t
                )
                norm_kp2d = norm_kp2d.unsqueeze(0).to(device, non_blocking=True)
                
                feature_t = torch.stack(list(hist['feature']), dim=0).unsqueeze(0).to(device, non_blocking=True)

                if hist['cached_inits'] is None or hist['cached_init_root'] is None:
                    hist['cached_inits'], hist['cached_init_root'] = build_wham_inits(
                        hist['init_data'],
                        norm_kp2d[0, 0].clone()
                    )

                if infer_interval > 1 and (frame_id % infer_interval) != 0 and hist['last_pred'] is not None:
                    last_verts_cam, last_smplx_params = hist['last_pred']
                    while not stop_event.is_set():
                        try:
                            wham_Q.put(
                                (
                                    frame_id,
                                    start_time,
                                    frame,
                                    True,
                                    last_verts_cam.copy(),
                                    track_id,
                                    {k: v.copy() for k, v in last_smplx_params.items()},
                                ),
                                timeout=0.1,
                            )
                            break
                        except Full:
                            continue
                    continue
                
                cam_angvel = torch.zeros((1, norm_kp2d.shape[1], 6), device=device)
                
                inits = hist['cached_inits']
                init_root = hist['cached_init_root']

                try:
                    _t_infer0 = time.perf_counter()
                    with torch.inference_mode():
                        with autocast_ctx():
                            pred = network(
                                norm_kp2d,
                                inits,
                                feature_t,
                                mask=mask,
                                init_root=init_root,
                                cam_angvel=cam_angvel,
                                return_y_up=True,
                                cam_intrinsics=intrinsics_batch,
                                bbox=bbox_t.unsqueeze(0).to(device, non_blocking=True),
                                res=res_batch,
                                refine_traj=False,
                                states=None
                            )
                    _t_infer_ms = (time.perf_counter() - _t_infer0) * 1000

                    verts_cam = extract_latest_vertices(pred).detach().cpu().numpy()
                    smplx_params = extract_latest_smplx_params(pred)
                    if frame_id % 10 == 0:
                        logger.info(f"[WHAM] infer_ms={_t_infer_ms:.1f}  frame={frame_id}")
                    hist['last_pred'] = (verts_cam.copy(), {k: v.copy() for k, v in smplx_params.items()})
                    while not stop_event.is_set():
                        try:
                            wham_Q.put((frame_id, start_time, frame, True, verts_cam, track_id, smplx_params), timeout=0.1)
                            break
                        except Full:
                            continue
                except Exception:
                    logger.exception(f"WHAM inference failed on frame {frame_id}")
                    while not stop_event.is_set():
                        try:
                            wham_Q.put((frame_id, start_time, frame, False, None, track_id, None), timeout=0.1)
                            break
                        except Full:
                            continue
            else:
                while not stop_event.is_set():
                    try:
                        wham_Q.put((frame_id, start_time, frame, False, None, None, None), timeout=0.1)
                        break
                    except Full:
                        continue
                if frame_id == 0 or frame_id % heartbeat_frames == 0:
                    logger.warning(f"[WHAM] frame={frame_id} skipped because no tracked person; wham_Q={safe_qsize(wham_Q)}")

    t1 = threading.Thread(target=lambda: guarded_thread("reader", reader_thread), daemon=True, name="reader")
    t2 = threading.Thread(target=lambda: guarded_thread("detector", detector_thread), daemon=True, name="detector")
    t3 = threading.Thread(target=lambda: guarded_thread("extractor", extractor_thread), daemon=True, name="extractor")
    t4 = threading.Thread(target=lambda: guarded_thread("wham", wham_thread), daemon=True, name="wham")
    terminal_listener = None

    if is_webcam:
        if webcam_capture_time > 0:
            logger.info(f"Webcam auto-stop enabled: {webcam_capture_time:.2f}s")
        else:
            terminal_listener = threading.Thread(
                target=lambda: guarded_thread("terminal-listener", terminal_key_listener),
                daemon=True,
                name="terminal-listener",
            )
    
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    if terminal_listener is not None:
        terminal_listener.start()
    if record_whamvideo:
        render_thread = threading.Thread(
            target=lambda: guarded_thread("wham-render", render_thread_worker),
            daemon=True,
            name='wham-render',
        )
        render_thread.start()
    

    
    logger.info("Pipeline started.")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # GMR State Setup.  Multi-robot mode shares the WHAM/human-motion
    # adaptation path, then runs one GMR state per selected robot.
    def _parse_robot_keys():
        raw = getattr(gmr_args, "robots", None) if gmr_args is not None else None
        if not raw:
            raw = getattr(gmr_args, "robot", "unitree_g1") if gmr_args is not None else "unitree_g1"
        keys = []
        seen = set()
        for part in str(raw).split(","):
            key = part.strip()
            if not key:
                continue
            low = key.lower()
            if low not in seen:
                keys.append(key)
                seen.add(low)
        return keys or [getattr(gmr_args, "robot", "unitree_g1")]

    robot_keys = _parse_robot_keys()

    def _new_gmr_state(robot_key, index):
        return {
            "robot": robot_key,
            "index": index,
            "lock": threading.Lock(),
            "worker_queue": None,
            "worker_thread": None,
            "worker_last_frame": -1,
            "worker_last_error": None,
            "worker_dropped": 0,
            "latest_qp": None,
            "latest_frame_id": -1,
            "latest_row": None,
            "last_ik_ms": 0.0,
            "last_post_ms": 0.0,
            "last_csv_ms": 0.0,
            "retarget": None,
            "postprocessor": None,
            "viewer": None,
            "renderer": None,
            "tcp_sender": None,
            "csv_file": None,
            "csv_writer": None,
            "csv_path": None,
            "metadata_path": None,
            "metadata_written": False,
            "csv_row_count": 0,
            "qpos_history": [],
            "prev_qpos_quat": None,
            "ik_residuals_acc": [],
            "ik_pos_error_acc": [],
        }

    gmr_states = {robot: _new_gmr_state(robot, i) for i, robot in enumerate(robot_keys)}
    gmr_state = gmr_states[robot_keys[0]]
    gmr_state.update({
        "tail_body_model": None,
        "tail_joint_names": None,
        "tail_parents": None,
        # WHAM trans_world resets per sliding window (cumsum starts from 0).
        # Integrate root velocity ourselves for continuous pelvis position.
        "global_pelvis_pos": None,       # integrated Z-up position (3,)
        "prev_raw_pelvis_pos": None,     # for fallback velocity estimation
        "prev_frame_id": None,           # for dt calculation
        # Root orientation tracking for continuity across WHAM windows.
        # WHAM root orientation also resets per window; clamp large jumps.
        "prev_pelvis_quat": None,        # xyzw quaternion
    })
    csv_flush_rows = max(1, int(os.environ.get("CSV_FLUSH_ROWS", "5")))
    csv_flush_interval = max(0.01, float(os.environ.get("CSV_FLUSH_INTERVAL_MS", "100")) / 1000.0)
    
    if gmr_args is not None:
        csv_root = getattr(gmr_args, "csv_root", None)
        for state in gmr_states.values():
            robot_key = state["robot"]
            if csv_root:
                csv_path = os.path.join(csv_root, robot_key, "live_motion.csv")
            elif len(gmr_states) == 1:
                csv_path = gmr_args.csv_path
            else:
                base_dir = os.path.dirname(gmr_args.csv_path) or "pkl_outputs/csv"
                csv_path = os.path.join(base_dir, robot_key, "live_motion.csv")

            csv_dir = os.path.dirname(csv_path)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
            state["csv_path"] = csv_path
            state["csv_file"] = open(csv_path, "w", newline="")
            state["csv_writer"] = csv.writer(state["csv_file"])
            state["last_csv_flush_time"] = time.time()
            logger.info(f"[GMR] CSV output opened: robot={robot_key} path={csv_path}")
        
    def tail_params_to_smplx_frame(params):
        _dev = torch.device(gmr_args.torch_device) if gmr_args and gmr_args.torch_device != "cpu" else torch.device("cpu")

        if gmr_state["tail_body_model"] is None:
            bm = smplx.create(
                str(SMPLX_FOLDER),
                "smplx",
                gender="neutral",
                use_pca=False,
            )
            bm = bm.to(_dev)
            gmr_state["tail_body_model"] = bm
            gmr_state["tail_joint_names"] = JOINT_NAMES[:len(bm.parents)]
            gmr_state["tail_parents"] = bm.parents

        bm = gmr_state["tail_body_model"]
        jnames = gmr_state["tail_joint_names"]
        parents = gmr_state["tail_parents"]

        bpose = np.asarray(params["body_pose"], dtype=np.float32).reshape(1, 63)
        rorient = np.asarray(params["global_orient_global"], dtype=np.float32).reshape(1, 3)
        trans = np.asarray(params["transl_global"], dtype=np.float32).reshape(1, 3)

        if gmr_args and gmr_args.coord_fix in ("auto", "yup_to_zup"):
            rorient, trans = _tail_apply_yup_to_zup(rorient, trans)

        betas = _align_tail_betas(params["betas"], bm)

        with torch.no_grad():
            s_out = bm(
                betas=torch.from_numpy(betas).float().to(_dev).view(1, -1),
                global_orient=torch.from_numpy(rorient).float().to(_dev),
                body_pose=torch.from_numpy(bpose).float().to(_dev),
                transl=torch.from_numpy(trans).float().to(_dev),
                left_hand_pose=torch.zeros(1, 45).float().to(_dev),
                right_hand_pose=torch.zeros(1, 45).float().to(_dev),
                jaw_pose=torch.zeros(1, 3).float().to(_dev),
                leye_pose=torch.zeros(1, 3).float().to(_dev),
                reye_pose=torch.zeros(1, 3).float().to(_dev),
                return_full_pose=True,
            )

        so = s_out.global_orient[0].detach().cpu().numpy()
        fp = s_out.full_pose[0].detach().cpu().numpy().reshape(-1, 3)
        sj = s_out.joints[0].detach().cpu().numpy()

        res = {}
        jorients = []
        for i, jn in enumerate(jnames):
            if i == 0: r = R.from_rotvec(so)
            else: r = jorients[parents[i]] * R.from_rotvec(fp[i].squeeze())
            jorients.append(r)
            q_xyzw = r.as_quat()
            res[jn] = (sj[i], q_xyzw[[3, 0, 1, 2]])  # xyzw -> wxyz

        return res, betas
        
    def init_gmr(betas):
        if gmr_state["retarget"] is not None: return
        hh = None
        if betas is not None and betas.size > 0:
            hh = 1.66 + 0.1 * float(betas[0])
            
        r_kwargs = dict(actual_human_height=hh, src_human="smplx", tgt_robot=gmr_args.robot)
        if gmr_args.robot_path is not None:
            r_kwargs["robot_path"] = gmr_args.robot_path
        
        gmr_state["retarget"] = GMR(**r_kwargs)
        # GMR_MAX_ITER: reduce from default 10 to speed up IK at the cost of
        # slightly lower convergence.  4–5 iterations are sufficient for
        # streaming use where the pose changes slowly between frames.
        expected_cols = _expected_csv_columns_for_robot(gmr_args.robot)
        model_nq = int(gmr_state["retarget"].model.nq)
        logger.info(
            f"[GMR] initialized robot={gmr_args.robot} model_nq={model_nq} "
            f"model_nv={int(gmr_state['retarget'].model.nv)} model_nu={int(gmr_state['retarget'].model.nu)} "
            f"expected_csv_cols={expected_cols if expected_cols is not None else '<unchecked>'} "
            f"xml={gmr_state['retarget'].xml_file}"
        )
        if expected_cols is not None and model_nq != expected_cols:
            raise RuntimeError(
                f"[GMR] Robot model qpos length mismatch: robot={gmr_args.robot} "
                f"model_nq={model_nq} expected_csv_cols={expected_cols} xml={gmr_state['retarget'].xml_file}"
            )

        default_gmr_max_iter = "20" if (gmr_args.robot or "").strip().lower() == "openloong" else "10"
        _gmr_max_iter = int(os.environ.get("GMR_MAX_ITER", default_gmr_max_iter))
        gmr_state["retarget"].max_iter = _gmr_max_iter
        logger.info(f"[GMR] max_iter={_gmr_max_iter}")
        gmr_state["postprocessor"] = OnlineQposPostprocessor(
            gmr_state["retarget"].xml_file,
            root_body_name=gmr_state["retarget"].robot_root_name,
            smooth_alpha=max(0.0, min(1.0, gmr_args.smooth_alpha)),
            height_adjust=gmr_args.height_adjust,
            root_origin_offset=gmr_args.root_origin_offset,
            torch_device=gmr_args.torch_device,
        )
        _write_unity_qpos_metadata(gmr_state)
        
        # Create MuJoCo viewer when needed: recording, screenshot capture, or TCP streaming
        _need_viewer = gmr_args.record_gmrvideo or capture_interval > 0 or (gmr_args.tcp and tcp_stream_gmr)
        _is_headless = not os.environ.get('DISPLAY')
        if _need_viewer:
            v_kws = dict(
                robot_type=gmr_args.robot,
                motion_fps=30.0,
                transparent_robot=0,
                record_video=gmr_args.record_gmrvideo,
                video_path=gmr_args.video_path,
            )
            gmr_window_w, gmr_window_h = gmr_window_size()
            v_kws["video_width"] = gmr_window_w
            v_kws["video_height"] = gmr_window_h
            v_kws["window_width"] = gmr_window_w
            v_kws["window_height"] = gmr_window_h
            logger.info(f"GMR viewer/window target size: {gmr_window_w}x{gmr_window_h} (input aspect {width / max(float(height), 1.0):.3f})")
            v_kws["camera_follow"] = getattr(gmr_args, "camera_follow", False) or getattr(gmr_args, "track", False)
            if hasattr(gmr_args, 'camera_lookat_height_offset'):
                v_kws["camera_lookat_height_offset"] = gmr_args.camera_lookat_height_offset
            if hasattr(gmr_args, 'camera_elevation'):
                v_kws["camera_elevation"] = gmr_args.camera_elevation
            if hasattr(gmr_args, 'camera_distance_scale'):
                v_kws["camera_distance_scale"] = gmr_args.camera_distance_scale
            if getattr(gmr_args, 'camera_azimuth', None) is not None:
                v_kws["camera_azimuth"] = gmr_args.camera_azimuth
            if getattr(gmr_args, 'robot_path', None) is not None:
                v_kws['robot_path'] = gmr_args.robot_path

            # On headless servers, use the minimal offscreen viewer (no GLFW needed)
            if _is_headless and not gmr_args.record_gmrvideo:
                try:
                    gmr_state["viewer"] = _HeadlessViewer(**v_kws)
                    logger.info(f"Initialized headless GMR viewer for {gmr_args.robot}")
                except Exception as e:
                    logger.error(f"Failed to initialize headless GMR viewer: {e}")
                    gmr_state["viewer"] = None

                if gmr_state["viewer"] is not None and ((gmr_args.tcp and tcp_stream_gmr) or capture_interval > 0):
                    try:
                        v = gmr_state["viewer"]
                        h, w, render_reason = gmr_renderer_size(capture_interval > 0)
                        gmr_state["renderer"] = mj.Renderer(v.model, height=h, width=w)
                        logger.info(f"Initialized GMR offscreen renderer ({w}x{h}) for {render_reason}")
                    except Exception as e:
                        logger.error(f"Failed to create offscreen renderer (MuJoCo needs EGL/GPU): {e}")
                        logger.error("  Try: export MUJOCO_GL=egl  or  pip install mujoco[egl]")
                        gmr_state["renderer"] = None
            else:
                if _is_headless and gmr_args.record_gmrvideo:
                    logger.warning("Video recording requires a display server; may fail on headless.")
                try:
                    gmr_state["viewer"] = RobotMotionViewer(**v_kws)
                    logger.info(f"Initialized GMR viewer for {gmr_args.robot}")

                    # Offscreen renderer for TCP streaming and screenshot capture
                    if (gmr_args.tcp and tcp_stream_gmr) or capture_interval > 0:
                        v = gmr_state["viewer"]
                        h, w, render_reason = gmr_renderer_size(capture_interval > 0)
                        gmr_state["renderer"] = mj.Renderer(v.model, height=h, width=w)
                        logger.info(f"Initialized GMR offscreen renderer ({w}x{h}) for {render_reason}")
                except Exception as e:
                    logger.error(f"Failed to initialize GMR viewer: {e}")
                    gmr_state["viewer"] = None

    def init_gmr_state(state, betas, create_viewer=None):
        _need_viewer = (
            bool(create_viewer) if create_viewer is not None
            else (gmr_args.record_gmrvideo or capture_interval > 0 or (gmr_args.tcp and tcp_stream_gmr))
        )
        if state["retarget"] is not None:
            if not _need_viewer or state.get("viewer") is not None:
                return
        else:
            robot_key = state["robot"]
            hh = None
            if betas is not None and betas.size > 0:
                hh = 1.66 + 0.1 * float(betas[0])

            r_kwargs = dict(actual_human_height=hh, src_human="smplx", tgt_robot=robot_key)
            if gmr_args.robot_path is not None and len(gmr_states) == 1:
                r_kwargs["robot_path"] = gmr_args.robot_path

            state["retarget"] = GMR(**r_kwargs)
            expected_cols = _expected_csv_columns_for_robot(robot_key)
            model_nq = int(state["retarget"].model.nq)
            logger.info(
                f"[GMR] initialized robot={robot_key} model_nq={model_nq} "
                f"model_nv={int(state['retarget'].model.nv)} model_nu={int(state['retarget'].model.nu)} "
                f"expected_csv_cols={expected_cols if expected_cols is not None else '<unchecked>'} "
                f"xml={state['retarget'].xml_file}"
            )
            if expected_cols is not None and model_nq != expected_cols:
                raise RuntimeError(
                    f"[GMR] Robot model qpos length mismatch: robot={robot_key} "
                    f"model_nq={model_nq} expected_csv_cols={expected_cols} xml={state['retarget'].xml_file}"
                )

            default_gmr_max_iter = "20" if (robot_key or "").strip().lower() == "openloong" else "10"
            _gmr_max_iter = int(os.environ.get("GMR_MAX_ITER", default_gmr_max_iter))
            state["retarget"].max_iter = _gmr_max_iter
            logger.info(f"[GMR] robot={robot_key} max_iter={_gmr_max_iter}")

            state["postprocessor"] = OnlineQposPostprocessor(
                state["retarget"].xml_file,
                root_body_name=state["retarget"].robot_root_name,
                smooth_alpha=max(0.0, min(1.0, gmr_args.smooth_alpha)),
                height_adjust=gmr_args.height_adjust,
                root_origin_offset=gmr_args.root_origin_offset,
                torch_device=gmr_args.torch_device,
            )
            _write_unity_qpos_metadata(state)

        if not _need_viewer:
            return

        robot_key = state["robot"]

        gmr_window_w, gmr_window_h = gmr_window_size()
        v_kws = dict(
            robot_type=robot_key,
            motion_fps=30.0,
            transparent_robot=0,
            record_video=gmr_args.record_gmrvideo and state["index"] == 0,
            video_path=gmr_args.video_path,
            video_width=gmr_window_w,
            video_height=gmr_window_h,
            window_width=gmr_window_w,
            window_height=gmr_window_h,
            camera_follow=getattr(gmr_args, "camera_follow", False) or getattr(gmr_args, "track", False),
        )
        if hasattr(gmr_args, "camera_lookat_height_offset"):
            v_kws["camera_lookat_height_offset"] = gmr_args.camera_lookat_height_offset
        if hasattr(gmr_args, "camera_elevation"):
            v_kws["camera_elevation"] = gmr_args.camera_elevation
        if hasattr(gmr_args, "camera_distance_scale"):
            v_kws["camera_distance_scale"] = gmr_args.camera_distance_scale
        if getattr(gmr_args, "camera_azimuth", None) is not None:
            v_kws["camera_azimuth"] = gmr_args.camera_azimuth
        if getattr(gmr_args, "robot_path", None) is not None and len(gmr_states) == 1:
            v_kws["robot_path"] = gmr_args.robot_path

        _is_headless = not os.environ.get("DISPLAY")
        try:
            if _is_headless and not (gmr_args.record_gmrvideo and state["index"] == 0):
                state["viewer"] = _HeadlessViewer(**v_kws)
                logger.info(f"Initialized headless GMR viewer for {robot_key}")
            else:
                state["viewer"] = RobotMotionViewer(**v_kws)
                logger.info(f"Initialized GMR viewer for {robot_key}")
        except Exception as e:
            logger.error(f"Failed to initialize GMR viewer for {robot_key}: {e}")
            state["viewer"] = None

        if state["viewer"] is not None and ((gmr_args.tcp and tcp_stream_gmr) or capture_interval > 0):
            try:
                v = state["viewer"]
                h, w, render_reason = gmr_renderer_size(capture_interval > 0)
                state["renderer"] = mj.Renderer(v.model, height=h, width=w)
                logger.info(f"Initialized GMR offscreen renderer ({w}x{h}) for {render_reason}, robot={robot_key}")
            except Exception as e:
                logger.error(f"Failed to create offscreen renderer for {robot_key}: {e}")
                state["renderer"] = None

    # TCP sender for streaming WHAM (streamId=0) and GMR (streamId=1) frames to Unity
    if gmr_args.tcp and (tcp_stream_wham or tcp_stream_gmr):
        tcp_sender = TcpStreamSender()
        for state in gmr_states.values():
            state["tcp_sender"] = tcp_sender
    else:
        for state in gmr_states.values():
            state["tcp_sender"] = None
        logger.info("TCP streaming disabled (use --tcp to enable)")

    frame_times = deque(maxlen=30)
    last_frame_time = time.time()
    render_error_count = 0
    frame_param_map = {}
    frame_timestamp_map = {}
    max_frame_id = -1

    # Per-component latency tracking (rolling 30-frame window)
    _t_gmr_ik   = deque(maxlen=30)   # GMR IK solve (ms)
    _t_gmr_post = deque(maxlen=30)   # post-processing (ms)
    _t_wham_q   = deque(maxlen=30)   # wham_Q.get wait time (ms)
    _t_adapt    = deque(maxlen=30)   # adaptation layer (ms)
    last_gmr_preview_render_time = 0.0
    gmr_worker_queue_size = max(1, int(os.environ.get("GMR_WORKER_QUEUE_SIZE", "1")))

    def _clone_smplx_frame(src):
        return {
            key: (
                np.asarray(value[0]).copy(),
                np.asarray(value[1]).copy(),
            )
            for key, value in src.items()
        }

    def _enqueue_latest_gmr_task(state, frame_id, sf, betas):
        q = state.get("worker_queue")
        if q is None:
            return

        task = (int(frame_id), _clone_smplx_frame(sf), None if betas is None else np.asarray(betas).copy())
        try:
            q.put_nowait(task)
            return
        except Full:
            pass

        try:
            q.get_nowait()
            with state["lock"]:
                state["worker_dropped"] += 1
        except Empty:
            pass

        try:
            q.put_nowait(task)
        except Full:
            with state["lock"]:
                state["worker_dropped"] += 1

    def _process_gmr_worker_frame(state, frame_id, sf, betas):
        robot_key = state["robot"]
        init_gmr_state(state, betas, create_viewer=False)

        _t0_ik = time.perf_counter()
        qp, ik_residuals = state["retarget"].retarget(sf)
        ik_ms = (time.perf_counter() - _t0_ik) * 1000

        _t0_post = time.perf_counter()
        qp = state["postprocessor"].process(qp)
        post_ms = (time.perf_counter() - _t0_post) * 1000

        if not os.environ.get("BENCH_DISABLE_QUAT_CLAMP", "0") == "1" and state.get("prev_qpos_quat") is not None:
            prev_q = state["prev_qpos_quat"]
            curr_q = np.asarray(qp[3:7], dtype=np.float64)
            if float(np.dot(curr_q, prev_q)) < 0.0:
                curr_q = -curr_q
            prev_xyzw = prev_q[[1, 2, 3, 0]]
            curr_xyzw = curr_q[[1, 2, 3, 0]]
            r_curr = R.from_quat(curr_xyzw)
            r_prev = R.from_quat(prev_xyzw)
            r_diff = r_curr * r_prev.inv()
            angle = float(np.linalg.norm(r_diff.as_rotvec()))
            max_angle = np.deg2rad(20.0)
            if angle > max_angle:
                blend = float(max_angle / angle)
                curr_q = prev_q + blend * (curr_q - prev_q)
                curr_q /= np.linalg.norm(curr_q)
                qp[3:7] = curr_q.astype(qp.dtype)
        state["prev_qpos_quat"] = np.asarray(qp[3:7], dtype=np.float64).copy()

        state["ik_residuals_acc"].append(ik_residuals)
        try:
            pos_err_mm = state["retarget"].decomposed_position_error_mm()
            state["ik_pos_error_acc"].append(pos_err_mm)
        except Exception:
            pass

        if str(robot_key).strip().lower() == "x02lite" and (state["csv_row_count"] == 0 or frame_id % heartbeat_frames == 0):
            _log_x02lite_qpos_health(qp, state.get("postprocessor"), frame_id, state)

        _t0_csv = time.perf_counter()
        row = _build_unity_csv_row(qp, robot_key, state["csv_path"], frame_id)
        writer = state.get("csv_writer")
        csv_fh = state.get("csv_file")
        if writer is not None:
            writer.writerow(row.tolist())
            state["csv_row_count"] += 1
            now_for_flush = time.time()
            if csv_fh is not None and (
                state["csv_row_count"] % csv_flush_rows == 0 or
                now_for_flush - state.get("last_csv_flush_time", 0.0) >= csv_flush_interval
            ):
                csv_fh.flush()
                state["last_csv_flush_time"] = now_for_flush
        csv_ms = (time.perf_counter() - _t0_csv) * 1000

        state["qpos_history"].append(qp.copy())
        with state["lock"]:
            state["latest_qp"] = qp.copy()
            state["latest_row"] = row.copy()
            state["latest_frame_id"] = int(frame_id)
            state["worker_last_frame"] = int(frame_id)
            state["worker_last_error"] = None
            state["last_ik_ms"] = ik_ms
            state["last_post_ms"] = post_ms
            state["last_csv_ms"] = csv_ms

        _t_gmr_ik.append(ik_ms)
        _t_gmr_post.append(post_ms)

        if state["csv_row_count"] == 1 or frame_id % heartbeat_frames == 0:
            logger.info(
                f"[GMR] wrote CSV row={state['csv_row_count']} frame={frame_id} robot={robot_key} "
                f"qpos_len={len(qp)} csv_cols={len(row)} csv={state['csv_path']} "
                f"ik_ms={ik_ms:.1f} post_ms={post_ms:.1f} csv_ms={csv_ms:.1f}"
            )

    def _gmr_worker_loop(state):
        robot_key = state["robot"]
        logger.info(f"[GMR] worker started robot={robot_key}")
        while not stop_event.is_set():
            try:
                task = state["worker_queue"].get(timeout=0.1)
            except Empty:
                continue
            if task is None:
                break
            frame_id, sf, betas = task
            try:
                _process_gmr_worker_frame(state, frame_id, sf, betas)
            except KeyboardInterrupt:
                terminal_stop_event.set()
                stop_event.set()
                break
            except Exception as e:
                with state["lock"]:
                    state["worker_last_error"] = str(e)
                logger.error(f"[GMR] worker error robot={robot_key} frame={frame_id}: {e}")
                if state.get("retarget") is None:
                    stop_event.set()
                    break
        logger.info(f"[GMR] worker stopped robot={robot_key}")

    for state in gmr_states.values():
        state["worker_queue"] = Queue(maxsize=gmr_worker_queue_size)
        worker = threading.Thread(
            target=_gmr_worker_loop,
            args=(state,),
            daemon=True,
            name=f"gmr-worker-{state['robot']}",
        )
        state["worker_thread"] = worker
        worker.start()

    # Register SIGINT handler so Ctrl+C sets stop flags gracefully instead of
    # interrupting PyTorch / MuJoCo native code mid-operation (which causes segfaults).
    def _on_sigint(signum, frame):
        logger.info("SIGINT received, requesting graceful shutdown...")
        terminal_stop_event.set()
        stop_event.set()

    prev_sigint = signal.signal(signal.SIGINT, _on_sigint)

    while True:
        try:
            _t0_q = time.perf_counter()
            res_data = wham_Q.get(timeout=0.1)
            _t_wham_q.append((time.perf_counter() - _t0_q) * 1000)
        except Empty:
            if stop_event.is_set():
                # If stop is requested from terminal, exit promptly instead of waiting
                # for all queued frames to drain (prevents webcam window freeze).
                if terminal_stop_event.is_set():
                    logger.info("Terminal stop requested, exiting WHAM main loop.")
                    break

                # If workers are all down and queue is empty, exit.
                if (not t1.is_alive()) and (not t2.is_alive()) and (not t3.is_alive()) and (not t4.is_alive()):
                    break
            continue

        if res_data is None:
            break
        frame_id, start_time, frame, success, verts_cam, pred_track_id, gvhmr_params = res_data
        max_frame_id = max(max_frame_id, frame_id)

        if terminal_stop_event.is_set():
            logger.info("Terminal stop requested, stop after current frame.")
            break

        if gvhmr_params is not None:
            frame_param_map[frame_id] = gvhmr_params
            frame_timestamp_map[frame_id] = float(start_time)
            
            # Process via GMR immediately in queue thread
            if gmr_args is not None:
                _t0_adapt = time.perf_counter()
                sf, bt = tail_params_to_smplx_frame(gvhmr_params)

                # Ablation switches (set via environment variables for benchmarking).
                # Normal usage: all disabled (0) → full stabilization active.
                _bench_no_vel  = os.environ.get("BENCH_DISABLE_VEL_INTEGRATION", "0") == "1"
                _bench_no_disp = os.environ.get("BENCH_DISABLE_DISP_THRESH", "0") == "1"
                _bench_no_quat = os.environ.get("BENCH_DISABLE_QUAT_CLAMP", "0") == "1"

                # WHAM resets trans_world (cumsum from zero) each sliding window
                # because we call with states=None.  Integrate root velocity
                # ourselves for a continuous pelvis position across windows.
                curr_raw = sf["pelvis"][0].astype(np.float64)

                if gmr_state["global_pelvis_pos"] is None:
                    gmr_state["global_pelvis_pos"] = curr_raw.copy()
                    gmr_state["prev_raw_pelvis_pos"] = curr_raw.copy()
                    gmr_state["prev_frame_id"] = frame_id
                elif not _bench_no_vel:
                    vel_root = gvhmr_params.get('vel_root')
                    root_mat = gvhmr_params.get('poses_root_world_mat')

                    if vel_root is not None and root_mat is not None:
                        # Rotate velocity from root-local to world frame (Y-up),
                        # then convert to Z-up.
                        vel_world_yup = root_mat.astype(np.float64) @ vel_root.astype(np.float64)
                        R_yup2zup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
                        vel_world_zup = vel_world_yup @ R_yup2zup.T
                        # vel_world is per-frame displacement; add directly.
                        gmr_state["global_pelvis_pos"] += vel_world_zup
                    elif not _bench_no_disp:
                        # Fallback: raw pelvis delta with tight threshold to
                        # reject WHAM origin resets (which produce large jumps).
                        raw_delta = curr_raw - gmr_state["prev_raw_pelvis_pos"]
                        max_step = float(os.environ.get("BENCH_DELTA_MAX", "0.15"))
                        if float(np.linalg.norm(raw_delta)) < max_step:
                            gmr_state["global_pelvis_pos"] += raw_delta

                    gmr_state["prev_raw_pelvis_pos"] = curr_raw.copy()
                    gmr_state["prev_frame_id"] = frame_id

                    # Shift all joint positions to the integrated global position.
                    offset = gmr_state["global_pelvis_pos"] - curr_raw
                    if float(np.linalg.norm(offset)) > 1e-6:
                        for jn in sf:
                            sf[jn] = (sf[jn][0].astype(np.float64) + offset, sf[jn][1])

                # ---- root orientation continuity across WHAM windows ----
                # WHAM resets root orientation each sliding window, causing
                # large yaw jumps.  Clamp the pelvis quaternion change per
                # frame to suppress these artifacts.
                pelvis_quat = np.asarray(sf["pelvis"][1], dtype=np.float64)
                if not _bench_no_quat and gmr_state["prev_pelvis_quat"] is not None:
                    prev_q = gmr_state["prev_pelvis_quat"]
                    # Ensure shortest path
                    if float(np.dot(pelvis_quat, prev_q)) < 0.0:
                        pelvis_quat = -pelvis_quat
                    # Angular difference via rotation vector
                    r_curr = R.from_quat(pelvis_quat[[1, 2, 3, 0]])  # wxyz -> xyzw
                    r_prev = R.from_quat(prev_q[[1, 2, 3, 0]])
                    r_diff = r_curr * r_prev.inv()
                    angle = float(np.linalg.norm(r_diff.as_rotvec()))
                    max_angle = np.deg2rad(15.0)  # max 15 deg / frame
                    if angle > max_angle:
                        # Clamp: LERP toward previous quaternion, then normalize.
                        blend = float(max_angle / angle)
                        pelvis_quat = prev_q + blend * (pelvis_quat - prev_q)
                        pelvis_quat /= np.linalg.norm(pelvis_quat)
                        sf["pelvis"] = (sf["pelvis"][0], pelvis_quat)
                gmr_state["prev_pelvis_quat"] = pelvis_quat.copy()

                _t_adapt.append((time.perf_counter() - _t0_adapt) * 1000)

                for state in gmr_states.values():
                    _enqueue_latest_gmr_task(state, frame_id, sf, bt)

                now_for_preview = time.time()
                render_gmr_preview_this_frame = (
                    tcp_stream_gmr and
                    (now_for_preview - last_gmr_preview_render_time) >= gmr_preview_interval
                )
                rendered_gmr_preview = False
                gmr_preview_frames = []
                if render_gmr_preview_this_frame or capture_interval > 0 or gmr_args.record_gmrvideo:
                    for robot_key, state in gmr_states.items():
                        try:
                            init_gmr_state(state, bt, create_viewer=True)
                            with state["lock"]:
                                qp = None if state.get("latest_qp") is None else state["latest_qp"].copy()
                            if qp is None:
                                continue

                            need_viewer_step = (
                                state["viewer"] is not None and
                                state.get("csv_row_count", 0) > max(0, int(gmr_args.viewer_warmup_frames)) and
                                (gmr_args.record_gmrvideo or capture_interval > 0 or render_gmr_preview_this_frame)
                            )
                            if not need_viewer_step:
                                continue

                            display_qp = qp.copy()
                            if len(gmr_states) > 1:
                                display_qp[0] += (state["index"] - (len(gmr_states) - 1) * 0.5) * 1.6
                            state["viewer"].step(
                                root_pos=display_qp[:3],
                                root_rot=display_qp[3:7],
                                dof_pos=display_qp[7:],
                                human_motion_data=None,
                                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                                show_human_body_name=False,
                                rate_limit=False,
                                follow_camera=(gmr_args.camera_follow or gmr_args.track),
                            )

                            gmr_renderer = state.get("renderer")
                            if tcp_stream_gmr and render_gmr_preview_this_frame and gmr_renderer is not None:
                                gmr_renderer.update_scene(state["viewer"].data, camera=state["viewer"].viewer.cam)
                                gmr_rgb = gmr_renderer.render()
                                gmr_preview_frames.append(cv2.cvtColor(gmr_rgb, cv2.COLOR_RGB2BGR))
                                rendered_gmr_preview = True
                        except Exception as e:
                            logger.error(f"GMR viewer error ({robot_key}): {e}")

                tcp = gmr_state.get("tcp_sender")
                if tcp_stream_gmr and gmr_preview_frames and tcp is not None:
                    if not tcp.connected:
                        tcp.connect()
                    try:
                        min_h = min(img.shape[0] for img in gmr_preview_frames)
                        fitted = []
                        for img in gmr_preview_frames:
                            if img.shape[0] != min_h:
                                w = max(1, int(img.shape[1] * (min_h / float(img.shape[0]))))
                                img = cv2.resize(img, (w, min_h), interpolation=cv2.INTER_AREA)
                            fitted.append(img)
                        gmr_bgr = np.concatenate(fitted, axis=1) if len(fitted) > 1 else fitted[0]
                        if tcp.send_frame(1, gmr_bgr) and not gmr_state.get("_tcp_gmr_logged", False):
                            logger.info(f"[TCP] Sent first GMR frame to Unity streamId=1 robots={','.join(robot_keys)}")
                            gmr_state["_tcp_gmr_logged"] = True
                    except Exception as e:
                        logger.error(f"GMR preview compose/send error: {e}")

                if rendered_gmr_preview:
                    last_gmr_preview_render_time = time.time()
        else:
            if frame_id == 0 or frame_id % heartbeat_frames == 0:
                logger.warning(
                    f"[Main] frame={frame_id} has no WHAM params; success={success} track_id={pred_track_id} "
                    f"queues read={safe_qsize(read_Q)} det={safe_qsize(det_Q)} ext={safe_qsize(ext_Q)} wham={safe_qsize(wham_Q)}"
                )
        
        curr_time = time.time()
        throughput_time = curr_time - last_frame_time
        last_frame_time = curr_time
        frame_times.append(throughput_time)
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
        
        # Calculate latency
        latency = curr_time - start_time

        if frame_id == 0 or frame_id % heartbeat_frames == 0:
            robot_rows_parts = []
            for key, state in gmr_states.items():
                q = state.get("worker_queue")
                with state["lock"]:
                    err = state.get("worker_last_error")
                    dropped = state.get("worker_dropped", 0)
                    latest_frame = state.get("latest_frame_id", -1)
                    ik_ms = state.get("last_ik_ms", 0.0)
                    csv_ms = state.get("last_csv_ms", 0.0)
                err_tag = "" if not err else f":err={str(err)[:48]}"
                robot_rows_parts.append(
                    f"{key}:csv={state.get('csv_row_count', 0)} latest={latest_frame} "
                    f"q={safe_qsize(q)} drop={dropped} ik={ik_ms:.1f} csv_ms={csv_ms:.1f}{err_tag}"
                )
            robot_rows = " | ".join(robot_rows_parts)
            mean_ik = float(np.mean(_t_gmr_ik)) if len(_t_gmr_ik) else 0.0
            mean_post = float(np.mean(_t_gmr_post)) if len(_t_gmr_post) else 0.0
            mean_adapt = float(np.mean(_t_adapt)) if len(_t_adapt) else 0.0
            mean_wham_wait = float(np.mean(_t_wham_q)) if len(_t_wham_q) else 0.0
            logger.info(
                f"[Heartbeat] frame={frame_id} wham_fps={avg_fps:.2f} latency={latency:.2f}s "
                f"queues read={safe_qsize(read_Q)} det={safe_qsize(det_Q)} ext={safe_qsize(ext_Q)} wham={safe_qsize(wham_Q)} "
                f"adapt_ms={mean_adapt:.1f} ik_ms={mean_ik:.1f} post_ms={mean_post:.1f} wham_wait_ms={mean_wham_wait:.1f} "
                f"gmr_preview_fps={gmr_preview_fps:.2f} {robot_rows}"
            )
        
        if record_whamvideo:
            if success and verts_cam is not None:
                # Render using WHAM Renderer
                if renderer is not None:
                    verts_tensor = torch.from_numpy(verts_cam).to(device=device, dtype=torch.float32)
                    # Render mesh expects bounding box update internally
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rendered_frame = renderer.render_mesh(verts_tensor, frame_rgb, colors=[0.2, 0.4, 0.8])
                        frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
                    except Exception:
                        render_error_count += 1
                        if render_error_count <= 5 or render_error_count % 50 == 0:
                            logger.exception(f"Render failed on frame {frame_id} (count={render_error_count})")
                cv2.putText(frame, f"Frame {frame_id} | FPS: {avg_fps:.1f} | Latency: {latency:.2f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame, f"Frame {frame_id} | FPS: {avg_fps:.1f} | Latency: {latency:.2f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            frame_out = sanitize_preview_frame(frame)
            # Feed the render thread for video encoding + preview window
            packet = (frame_out.copy(), float(curr_time))
            try:
                render_Q.put(packet, timeout=0.01)
            except Full:
                try:
                    _ = render_Q.get_nowait()
                    render_drop_count += 1
                except Empty:
                    pass
                try:
                    render_Q.put_nowait(packet)
                except Full:
                    render_drop_count += 1

        # Screenshot capture at specified intervals
        if capture_interval > 0 and frame_id % capture_interval == 0 and cap_orig is not None:
            capture_id = f"{frame_id:06d}"
            # 1) Original video frame (seek in source video)
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret_orig, orig_frame = cap_orig.read()
            if ret_orig and orig_frame is not None:
                cv2.imwrite(os.path.join(capture_dir, "orig", f"{capture_id}.png"), orig_frame)
            # 2) WHAM visualization frame
            if success and frame is not None:
                cv2.imwrite(os.path.join(capture_dir, "wham", f"{capture_id}.png"), frame)
            # 3) MuJoCo robot frame via offscreen renderer
            _gmr_rend = gmr_state.get("renderer")
            _gmr_view = gmr_state.get("viewer")
            _gmr_render_failures = gmr_state.setdefault("_render_failures", 0)
            if _gmr_rend is not None and _gmr_view is not None:
                try:
                    _gmr_rend.update_scene(_gmr_view.data, camera=_gmr_view.viewer.cam)
                    gmr_rgb = _gmr_rend.render()
                    gmr_bgr = cv2.cvtColor(gmr_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(capture_dir, "mujoco", f"{capture_id}.png"), gmr_bgr)
                except Exception as e:
                    _gmr_render_failures += 1
                    gmr_state["_render_failures"] = _gmr_render_failures
                    if _gmr_render_failures <= 3:
                        logger.warning(f"MuJoCo screenshot render failed (attempt {_gmr_render_failures}): {e}")
            if frame_id % (capture_interval * 5) == 0:
                logger.info(f"[Capture] saved frame {capture_id} to {capture_dir}")

        # Send WHAM frame to Unity via TCP (streamId=0)
        tcp = gmr_state.get("tcp_sender")
        if tcp_stream_wham and tcp is not None and not tcp.connected:
            tcp.connect()
        if tcp_stream_wham and tcp is not None and frame is not None:
            if tcp.send_frame(0, frame) and not gmr_state.get("_tcp_wham_logged", False):
                logger.info("[TCP] Sent first WHAM frame to Unity streamId=0")
                gmr_state["_tcp_wham_logged"] = True

        if frame_id % 10 == 0:
            _q_ms   = sum(_t_wham_q)   / len(_t_wham_q)   if _t_wham_q   else 0
            _ad_ms  = sum(_t_adapt)    / len(_t_adapt)    if _t_adapt    else 0
            _ik_ms  = sum(_t_gmr_ik)   / len(_t_gmr_ik)   if _t_gmr_ik   else 0
            _po_ms  = sum(_t_gmr_post) / len(_t_gmr_post) if _t_gmr_post else 0
            # Average IK residuals over the last 10-frame window
            _ikr1 = 0.0; _ikr2 = 0.0; _pos_err = 0.0
            acc = gmr_state.get("ik_residuals_acc", [])
            if acc:
                _ikr1 = float(np.mean([r[0] for r in acc]))
                _ikr2 = float(np.mean([r[1] for r in acc]))
            pos_acc = gmr_state.get("ik_pos_error_acc", [])
            if pos_acc:
                _pos_err = float(np.mean(pos_acc))
            # Reset accumulators
            gmr_state["ik_residuals_acc"] = []
            gmr_state["ik_pos_error_acc"] = []
            # Format the IK residual tag for benchmark parsing
            logger.info(f"[IK] res1={_ikr1:.4f}  res2={_ikr2:.4f}  pos_err_mm={_pos_err:.2f}")
            logger.info(
                f"Frame {frame_id} | FPS: {avg_fps:.1f} | Latency: {latency:.3f}s | "
                f"[Q-wait:{_q_ms:.1f}ms  Adapt:{_ad_ms:.1f}ms  IK:{_ik_ms:.1f}ms  Post:{_po_ms:.1f}ms]"
            )
            
    if gmr_args is not None:
        stop_event.set()
        for state in gmr_states.values():
            q = state.get("worker_queue")
            if q is None:
                continue
            try:
                q.put_nowait(None)
            except Full:
                try:
                    q.get_nowait()
                except Empty:
                    pass
                try:
                    q.put_nowait(None)
                except Full:
                    pass

        worker_join_timeout = max(0.5, float(os.environ.get("GMR_WORKER_JOIN_TIMEOUT", "3.0")))
        for state in gmr_states.values():
            worker = state.get("worker_thread")
            if worker is not None:
                worker.join(timeout=worker_join_timeout)
                if worker.is_alive():
                    logger.warning(f"[GMR] worker did not exit in time robot={state['robot']}")

        for state in gmr_states.values():
            csv_fh = state.get("csv_file")
            if csv_fh is not None:
                try:
                    csv_fh.flush()
                    csv_fh.close()
                except Exception:
                    pass

        from scripts.smplx_to_robot_stream import write_motion_pkl
        primary_history = gmr_state.get("qpos_history", [])
        if len(primary_history) > 0:
            logger.info(f"Saving motion PKL to {gmr_args.save_path}")
            try:
                write_motion_pkl(
                    gmr_args.save_path,
                    primary_history,
                    fps=30.0,
                    xml_file=gmr_state["retarget"].xml_file,
                    root_body_name=gmr_state["retarget"].robot_root_name,
                    torch_device=gmr_args.torch_device,
                )
            except Exception as e:
                logger.error(f"Failed to write motion pkl: {e}")
        if cap_orig is not None:
            # Log capture summary
            _dirs = {"orig": 0, "wham": 0, "mujoco": 0}
            for _sub in _dirs:
                _p = os.path.join(capture_dir, _sub)
                if os.path.isdir(_p):
                    _dirs[_sub] = len([f for f in os.listdir(_p) if f.endswith(".png")])
            logger.info(f"[Capture] summary: orig={_dirs['orig']} wham={_dirs['wham']} mujoco={_dirs['mujoco']}")
            if _dirs["mujoco"] == 0 and (gmr_state.get("renderer") is None):
                logger.warning("[Capture] No MuJoCo frames captured — renderer unavailable (try: export MUJOCO_GL=egl)")
            elif _dirs["mujoco"] == 0:
                logger.warning(f"[Capture] No MuJoCo frames captured — render failures={gmr_state.get('_render_failures', 0)}")
            cap_orig.release()
        for state in gmr_states.values():
            if state.get("viewer") is not None:
                try:
                    state["viewer"].close()
                except Exception:
                    pass
        if gmr_state.get("tcp_sender") is not None:
            try:
                gmr_state["tcp_sender"].close()
                logger.info("[TCP] Disconnected from Unity")
            except Exception:
                pass
        if tail_writer is not None:
            tail_writer.close(frame_count=len(frame_param_map))

    request_stop()
    join_timeout = max(1.0, float(os.environ.get('WHAM_THREAD_JOIN_TIMEOUT', '3.0')))
    for t, name in ((t1, 'reader'), (t2, 'detector'), (t3, 'extractor'), (t4, 'wham')):
        t.join(timeout=join_timeout)
        if t.is_alive():
            if terminal_stop_event.is_set():
                logger.info(f"{name} thread still running during fast-stop; continue shutdown.")
            else:
                logger.warning(f"{name} thread did not exit in time; continue shutdown.")
    if terminal_listener is not None and terminal_listener.is_alive():
        stop_event.set()
        terminal_listener.join(timeout=0.5)

    if render_thread is not None:
        render_thread.join(timeout=join_timeout)
        if render_thread.is_alive():
            logger.warning("render thread did not exit in time; continue shutdown.")
        if render_drop_count > 0:
            logger.info(f"Render queue dropped {render_drop_count} frame(s) to keep low latency.")

    with thread_failure_lock:
        fatal_thread_message = thread_failure.get("message")
    if fatal_thread_message:
        if torch.cuda.is_available():
            logger.info(f"[VRAM] peak_allocated_mb={torch.cuda.max_memory_allocated()/1024/1024:.0f}")
        raise RuntimeError(fatal_thread_message)

    if len(frame_param_map) == 0 or max_frame_id < 0:
        logger.warning("No valid WHAM predictions were collected. Skip GMR SMPL-X npz save.")
    else:
        full_frame_ids = np.arange(max_frame_id + 1, dtype=np.int64)

        first_valid = None
        for fid in full_frame_ids:
            if int(fid) in frame_param_map:
                first_valid = frame_param_map[int(fid)]
                break

        if first_valid is None:
            logger.warning("No valid WHAM parameters were collected. Skip GMR SMPL-X npz save.")
            cap.release()
            if torch.cuda.is_available():
                logger.info(f"[VRAM] peak_allocated_mb={torch.cuda.max_memory_allocated()/1024/1024:.0f}")
            logger.info("Stream processing finished.")
            return

        filled_body_pose = []
        filled_betas = []
        filled_orient_global = []
        filled_transl_global = []

        prev = first_valid
        for fid in full_frame_ids:
            cur = frame_param_map.get(int(fid), None)
            if cur is None:
                cur = prev
            else:
                prev = cur

            filled_body_pose.append(cur['body_pose'])
            filled_betas.append(cur['betas'])
            filled_orient_global.append(cur['global_orient_global'])
            filled_transl_global.append(cur['transl_global'])

        n_frames = len(full_frame_ids)

        body_pose_arr = np.stack(filled_body_pose, axis=0).astype(np.float32)
        betas_arr = np.stack(filled_betas, axis=0).astype(np.float32)
        orient_global_arr = np.stack(filled_orient_global, axis=0).astype(np.float32)
        transl_global_arr = np.stack(filled_transl_global, axis=0).astype(np.float32)

        # GMR loads only the first-frame betas from GVHMR files.
        # Stabilize betas across frames to avoid using a noisy warm-up frame.
        betas_ref = np.median(betas_arr, axis=0).astype(np.float32)
        betas_arr = np.repeat(betas_ref[None, :], n_frames, axis=0)

        # Save GMR-native SMPL-X npz for scripts/smplx_to_robot.py.
        gmr_smplx_npz = os.path.join(output_dir, 'gmr_smplx_results.npz')
        output_fps = estimate_fps_from_timestamps(
            [frame_timestamp_map.get(int(fid), np.nan) for fid in full_frame_ids],
            fallback_fps=float(fps),
        )
        np.savez(
            gmr_smplx_npz,
            pose_body=body_pose_arr,
            root_orient=orient_global_arr,
            trans=transl_global_arr,
            betas=betas_ref,
            mocap_frame_rate=np.array(float(output_fps), dtype=np.float32),
            gender=np.array('neutral'),
            coord_system=np.array('y-up'),
            source=np.array('wham_stream_mt'),
        )
        logger.info(f"Saved GMR SMPL-X npz to {gmr_smplx_npz} (frames={n_frames}, fps={output_fps:.2f})")

    cap.release()
    if torch.cuda.is_available():
        logger.info(f"[VRAM] peak_allocated_mb={torch.cuda.max_memory_allocated()/1024/1024:.0f}")
    logger.info("Stream processing finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='examples/drone_video.mp4')
    parser.add_argument('--output_dir', type=str, default='output/stream_demo')
    parser.add_argument('--record_whamvideo', action='store_true',
                        help='Enable WHAM mesh rendering and write WHAM output video.')
    parser.add_argument('--record_video', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--rotate', type=int, default=None, choices=[0, 90, 180, 270],
                        help='Optional manual clockwise rotation for input frames.')
    parser.add_argument('--stream_npz_dir', type=str, default=None,
                        help='Optional directory to write append-only SMPL-X tail stream (stream_tail.pkl).')
    parser.add_argument('--time', type=float, default=0.0,
                        help='When --video=0 (webcam): >0 auto-stop after this many seconds; 0 waits for terminal q/ESC.')
    
    # GMR Args
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--robots", default=None,
                        help="Comma-separated robot keys for multi-robot GMR. Defaults to --robot.")
    parser.add_argument("--robot_path", type=str, default=None)
    parser.add_argument("--torch_device", default="cpu")
    parser.add_argument("--coord_fix", default="yup_to_zup")
    parser.add_argument("--save_path", type=str, default="pkl_outputs/live_motion.pkl")
    parser.add_argument("--csv_path", type=str, default="pkl_outputs/csv/live_motion.csv")
    parser.add_argument("--csv_root", type=str, default=None,
                        help="Root directory for multi-robot CSV output: <csv_root>/<robot>/live_motion.csv.")
    parser.add_argument("--record_gmrvideo", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/live_stream_robot.mp4", help="GMR video path")
    parser.add_argument("--viewer_warmup_frames", type=int, default=12)
    parser.add_argument("--camera_follow", action="store_true")
    parser.add_argument("--no-camera_follow", dest="camera_follow", action="store_false")
    parser.set_defaults(camera_follow=False)
    parser.add_argument("--track", action="store_true", help="Alias for --camera_follow (GMR view follows robot)")
    parser.add_argument("--no-track", dest="track", action="store_false")
    parser.set_defaults(track=False)
    parser.add_argument("--tcp", action="store_true", help="Enable TCP streaming to Unity (default: off)")
    parser.add_argument("--no-tcp", dest="tcp", action="store_false")
    parser.add_argument("--no_tcp", dest="tcp", action="store_false")
    parser.set_defaults(tcp=False)
    parser.add_argument("--camera_lookat_height_offset", type=float, default=0.50)
    parser.add_argument("--camera_elevation", type=float, default=-28.0)
    parser.add_argument("--camera_distance_scale", type=float, default=1.70)
    parser.add_argument("--camera_azimuth", type=float, default=None)
    parser.add_argument("--smooth_alpha", type=float, default=0.35)
    parser.add_argument("--height_adjust", action="store_true")
    parser.add_argument("--no-height_adjust", dest="height_adjust", action="store_false")
    parser.set_defaults(height_adjust=True)
    parser.add_argument("--root_origin_offset", action="store_true")
    parser.add_argument("--no-root_origin_offset", dest="root_origin_offset", action="store_false")
    parser.set_defaults(root_origin_offset=False)
    parser.add_argument("--capture_interval", type=int, default=0,
                        help="Save screenshots every N frames (0=disabled). "
                             "Saves original video, WHAM viz, and MuJoCo robot frames.")
    parser.add_argument("--capture_dir", type=str, default="results/screenshots",
                        help="Directory for captured screenshots (default: results/screenshots)")
    parser.add_argument("--poll_interval", type=float, default=0.05)
    parser.add_argument("--idle_timeout", type=float, default=0.0)
    parser.add_argument("--done_grace_sec", type=float, default=2.0)
    parser.add_argument("--viewer_ready_timeout_sec", type=float, default=8.0)
    parser.add_argument("--viewer_thread_join_timeout_sec", type=float, default=10.0)
    
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    # `--record_video` is a legacy alias that made disabled HUD recording
    # unexpectedly write WHAM videos through old run.ps1 defaults. Keep the
    # parser for compatibility, but make explicit WHAM/GMR flags authoritative.
    record_whamvideo = bool(args.record_whamvideo)



    run_stream_mt(
        cfg,
        args.video,
        args.output_dir,
        rotate_deg=args.rotate,
        record_whamvideo=record_whamvideo,
        capture_time=max(0.0, float(args.time)),
        gmr_args=args,
        stream_npz_dir=getattr(args, 'stream_npz_dir', None),
        capture_interval=args.capture_interval,
        capture_dir=args.capture_dir,
    )
