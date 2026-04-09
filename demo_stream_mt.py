import argparse
import time
import os
import cv2
import sys
import select
import torch
import numpy as np
from loguru import logger
import threading
from queue import Queue, Full, Empty
from contextlib import nullcontext

try:
    import termios
    import tty
except Exception:
    termios = None
    tty = None

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
    stream_npz_dir=None,
    capture_time=0.0,
):
    os.makedirs(output_dir, exist_ok=True)
    device = cfg.DEVICE.lower()
    is_cuda_device = str(device).startswith('cuda') and torch.cuda.is_available()
    use_amp = env_flag('WHAM_USE_AMP', True) and is_cuda_device
    detect_interval = max(1, int(os.environ.get('WHAM_DETECT_INTERVAL', '2')))
    infer_interval = max(1, int(os.environ.get('WHAM_INFER_INTERVAL', '1')))
    stream_seq_len = max(4, int(os.environ.get('WHAM_STREAM_SEQ_LEN', str(STREAM_SEQ_LEN))))

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
    is_webcam = video_path == '0' or video_path == 0
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
    keypoints_normalizer = Normalizer(cfg)
    
    if is_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Faild to load video file {video_path}'

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
    logger.info(f"Effective input size: {width}x{height} (scale={input_scale:.3f})")

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
    read_Q = Queue(maxsize=10)
    det_Q = Queue(maxsize=10)
    ext_Q = Queue(maxsize=10)
    wham_Q = Queue(maxsize=10)
    render_Q = Queue(maxsize=max(8, int(os.environ.get('WHAM_RENDER_QUEUE_SIZE', '120'))))
    render_thread = None
    render_drop_count = 0

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

        return {
            'body_pose': body_pose_aa.detach().cpu().numpy(),
            'betas': betas.detach().cpu().numpy(),
            'global_orient_global': root_world_aa.detach().cpu().numpy(),
            'transl_global': trans_world.detach().cpu().numpy(),
        }

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

            need_detect = (frame_id - last_detect_frame) >= detect_interval or last_track_id is None
            if need_detect:
                detector.track(frame, fps, 1000)
                poses = detector.pose_results_last

                if len(poses) > 0:
                    valid_poses = []
                    for pose in poses:
                        valid = (pose['keypoints'][:, -1] > 0.3).sum()
                        if valid >= 6:
                            valid_poses.append(pose)

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
            else:
                while not stop_event.is_set():
                    try:
                        ext_Q.put((frame_id, start_time, frame, None, None, None, None, None), timeout=0.1)
                        break
                    except Full:
                        continue
                
    def build_wham_inits(init_data_hist, first_norm_kp2d):
        with torch.inference_mode():
            init_output = smpl.get_output(
                global_orient=init_data_hist['init_global_orient'].to(device),
                body_pose=init_data_hist['init_body_pose'].to(device),
                betas=init_data_hist['init_betas'].to(device),
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

                    verts_cam = extract_latest_vertices(pred).detach().cpu().numpy()
                    smplx_params = extract_latest_smplx_params(pred)
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

    t1 = threading.Thread(target=reader_thread, daemon=True)
    t2 = threading.Thread(target=detector_thread, daemon=True)
    t3 = threading.Thread(target=extractor_thread, daemon=True)
    t4 = threading.Thread(target=wham_thread, daemon=True)
    terminal_listener = None

    if is_webcam:
        if webcam_capture_time > 0:
            logger.info(f"Webcam auto-stop enabled: {webcam_capture_time:.2f}s")
        else:
            terminal_listener = threading.Thread(target=terminal_key_listener, daemon=True)
    
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    if terminal_listener is not None:
        terminal_listener.start()
    if record_whamvideo:
        render_thread = threading.Thread(target=render_thread_worker, daemon=True, name='wham-render')
        render_thread.start()
    

    
    logger.info("Pipeline started.")
    
    frame_times = deque(maxlen=30)
    last_frame_time = time.time()
    render_error_count = 0
    frame_param_map = {}
    frame_timestamp_map = {}
    max_frame_id = -1
    
    while True:
        try:
            res_data = wham_Q.get(timeout=0.1)
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
            if tail_writer is not None:
                tail_writer.write_record(frame_id, start_time, gvhmr_params)
        
        curr_time = time.time()
        throughput_time = curr_time - last_frame_time
        last_frame_time = curr_time
        frame_times.append(throughput_time)
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
        
        # Calculate latency
        latency = curr_time - start_time
        
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

        if frame_id % 10 == 0:
            logger.info(f"Frame {frame_id} processed | FPS: {avg_fps:.1f} | Latency: {latency:.3f} s")
            
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
    logger.info("Stream processing finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='examples/demo_video.mp4')
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
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    record_whamvideo = bool(args.record_whamvideo or args.record_video)

    run_stream_mt(
        cfg,
        args.video,
        args.output_dir,
        rotate_deg=args.rotate,
        record_whamvideo=record_whamvideo,
        stream_npz_dir=args.stream_npz_dir,
        capture_time=max(0.0, float(args.time)),
    )
