import argparse
import time
import os
import cv2
import sys
import select
import torch
import numpy as np
from loguru import logger
from collections import defaultdict
import threading
from queue import Queue, Full, Empty

try:
    import termios
    import tty
except Exception:
    termios = None
    tty = None

from configs.config import get_cfg_defaults
from lib.vis.renderer import Renderer
import imageio
from collections import deque
from lib.data.utils.normalizer import Normalizer
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.utils.kp_utils import root_centering
import torchvision.transforms as transforms
from lib.models.preproc.backbone.utils import process_image

STREAM_SEQ_LEN = 16

def xyxy_to_cxcys(bbox, s_factor=1.2):
    cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200 * s_factor
    return np.array([cx, cy, scale])

def cxcys_to_bbox(cx, cy, scale):
    w = h = scale * 200.0
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

def rotate_frame_bgr(frame, rotate_deg):
    if rotate_deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate_deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotate_deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
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


def run_stream_mt(
    cfg,
    video_path,
    output_dir,
    rotate_deg=None,
    record_whamvideo=False,
    stream_npz_dir=None,
    stream_chunk_size=16,
    capture_time=0.0,
):
    os.makedirs(output_dir, exist_ok=True)
    device = cfg.DEVICE.lower()
    root_dir = os.path.abspath(os.path.dirname(__file__))
    is_webcam = video_path == '0' or video_path == 0
    webcam_capture_time = max(0.0, float(capture_time))
    webcam_start_ts = None
    
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
    first_frame = probe
    if is_webcam:
        webcam_start_ts = time.time()
    height, width = first_frame.shape[:2]
    
    from lib.utils.imutils import compute_cam_intrinsics
    res = torch.tensor([width, height]).float()
    intrinsics = compute_cam_intrinsics(res)

    renderer = None
    writer = None
    wham_video_path = os.path.join(output_dir, 'output.mp4')
    writer_buffer = []
    writer_buffer_ts = []
    if record_whamvideo:
        focal_length = (width ** 2 + height ** 2) ** 0.5
        renderer = Renderer(width, height, focal_length, device, smpl.faces)
        if not is_webcam:
            writer = imageio.get_writer(
                wham_video_path,
                fps=float(fps),
                mode='I',
                format='FFMPEG',
                macro_block_size=1,
            )
        else:
            logger.info("Webcam WHAM writer will use measured runtime FPS for output video.")

    if stream_npz_dir is not None:
        os.makedirs(stream_npz_dir, exist_ok=True)
        # Clear stale chunks and done flag from previous runs.
        for name in os.listdir(stream_npz_dir):
            if name.startswith('chunk_') and name.endswith('.npz'):
                try:
                    os.remove(os.path.join(stream_npz_dir, name))
                except OSError:
                    pass
        done_flag = os.path.join(stream_npz_dir, 'stream_done.flag')
        if os.path.exists(done_flag):
            try:
                os.remove(done_flag)
            except OSError:
                pass


    # Queues for pipeline
    read_Q = Queue(maxsize=10)
    det_Q = Queue(maxsize=10)
    ext_Q = Queue(maxsize=10)
    wham_Q = Queue(maxsize=10)

    stream_chunk = []
    stream_chunk_index = 0
    stable_stream_betas = None

    def ensure_webcam_writer(force=False):
        nonlocal writer, writer_buffer, writer_buffer_ts
        if (not record_whamvideo) or (not is_webcam):
            return
        if writer is not None:
            return
        if len(writer_buffer) == 0:
            return
        # Wait for enough samples to estimate stable FPS unless we are forcing a flush.
        if (not force) and len(writer_buffer) < 15:
            return

        est_fps = estimate_fps_from_timestamps(writer_buffer_ts, fallback_fps=max(1.0, float(fps)))
        writer = imageio.get_writer(
            wham_video_path,
            fps=float(est_fps),
            mode='I',
            format='FFMPEG',
            macro_block_size=1,
        )
        for f in writer_buffer:
            writer.append_data(f)
        logger.info(f"Initialized webcam WHAM writer at {est_fps:.2f} FPS.")
        writer_buffer = []
        writer_buffer_ts = []

    def flush_stream_chunk(force=False):
        nonlocal stream_chunk_index, stream_chunk, stable_stream_betas
        if stream_npz_dir is None:
            return
        if len(stream_chunk) == 0:
            return
        if (not force) and (len(stream_chunk) < max(1, int(stream_chunk_size))):
            return

        frame_ids = np.array([int(item[0]) for item in stream_chunk], dtype=np.int64)
        frame_ids.sort()

        frame_dict = {int(item[0]): (item[1], float(item[2])) for item in stream_chunk}
        frame_data = []
        frame_ts = []
        for fid in frame_ids:
            data_i, ts_i = frame_dict[int(fid)]
            frame_data.append(data_i)
            frame_ts.append(ts_i)

        pose_body = np.stack([d['body_pose'] for d in frame_data], axis=0).astype(np.float32)
        root_orient = np.stack([d['global_orient_global'] for d in frame_data], axis=0).astype(np.float32)
        trans = np.stack([d['transl_global'] for d in frame_data], axis=0).astype(np.float32)
        if stable_stream_betas is None:
            stable_stream_betas = np.median(
                np.stack([d['betas'] for d in frame_data], axis=0).astype(np.float32), axis=0
            ).astype(np.float32)
        betas = stable_stream_betas.copy()
        chunk_fps = estimate_fps_from_timestamps(frame_ts, fallback_fps=float(fps))

        chunk_name = f"chunk_{stream_chunk_index:06d}_{int(frame_ids[0]):06d}_{int(frame_ids[-1]):06d}.npz"
        chunk_path = os.path.join(stream_npz_dir, chunk_name)
        tmp_chunk_path = chunk_path + '.tmp'
        with open(tmp_chunk_path, 'wb') as f:
            np.savez(
                f,
                frame_ids=frame_ids,
                pose_body=pose_body,
                root_orient=root_orient,
                trans=trans,
                betas=betas,
                mocap_frame_rate=np.array(float(chunk_fps), dtype=np.float32),
                gender=np.array('neutral'),
                coord_system=np.array('y-up'),
                source=np.array('wham_stream_mt'),
            )
        os.replace(tmp_chunk_path, chunk_path)
        logger.info(f"Stream chunk saved: {chunk_path} ({len(frame_ids)} frames, fps={chunk_fps:.2f})")
        stream_chunk = []
        stream_chunk_index += 1

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
        # Wake up any thread blocked on queue.get(). If a queue is full,
        # drop one item to make room for the sentinel so stop can propagate.
        for q in (read_Q, det_Q, ext_Q, wham_Q):
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
    
    def reader_thread():
        frame_id = 0
        if first_frame is not None:
            read_Q.put((frame_id, time.time(), first_frame))
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
            read_Q.put((frame_id, time.time(), frame))
            frame_id += 1
        read_Q.put(None)
    
    def detector_thread():
        detector.initialize_tracking()
        selected_track_id = None
        while not stop_event.is_set():
            data = read_Q.get()
            if data is None: det_Q.put(None); break
            frame_id, start_time, frame = data
            
            # 内部调用推理
            detector.track(frame, fps, 1000)
            
            # 使用最新的一帧结果
            poses = detector.pose_results_last
            
            # 我们只追踪最主主体
            track_id = None
            kp2d = None
            bbox_c = None
            
            if len(poses) > 0:
                valid_poses = []
                for pose in poses:
                    valid = (pose['keypoints'][:, -1] > 0.3).sum()
                    if valid >= 6:
                        valid_poses.append(pose)

                best_pose = None
                if len(valid_poses) > 0:
                    # Prefer previously selected track to avoid identity switching jitter.
                    if selected_track_id is not None:
                        for pose in valid_poses:
                            if pose['track_id'] == selected_track_id:
                                best_pose = pose
                                break

                    # Fallback: choose largest visible person.
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
            
            det_Q.put((frame_id, start_time, frame, track_id, kp2d, bbox_c))
            
    def extractor_thread():
        while not stop_event.is_set():
            data = det_Q.get()
            if data is None: ext_Q.put(None); break
            frame_id, start_time, frame, track_id, kp2d, bbox_c = data
            
            if track_id is not None:
                cx, cy, scale = bbox_c
                norm_img, _ = process_image(frame[..., ::-1], [cx, cy], scale, 256, 256)
                norm_img = torch.from_numpy(norm_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    feature = extractor.model(norm_img, encode=True)
                
                # Recompute init from current frame to avoid stale pose lock.
                dummy_tracking = {track_id: {}}
                with torch.no_grad():
                    extractor.predict_init(norm_img, dummy_tracking, track_id, flip_eval=False)
                init_data = dummy_tracking[track_id]
                
                ext_Q.put((frame_id, start_time, frame, track_id, kp2d, bbox_c, feature, init_data))
            else:
                ext_Q.put((frame_id, start_time, frame, None, None, None, None, None))
                
    def wham_thread():
        temporal_history = {}

        while not stop_event.is_set():
            data = ext_Q.get()
            if data is None: wham_Q.put(None); break
            frame_id, start_time, frame, track_id, kp2d, bbox_c, feature, init_data = data
            
            if track_id is not None:
                if track_id not in temporal_history:
                    temporal_history[track_id] = {
                        'kp2d': deque(maxlen=STREAM_SEQ_LEN),
                        'bbox': deque(maxlen=STREAM_SEQ_LEN),
                        'feature': deque(maxlen=STREAM_SEQ_LEN),
                        'init_data': init_data,
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
                norm_kp2d = norm_kp2d.unsqueeze(0).to(device)
                
                feature_t = torch.stack(list(hist['feature']), dim=0).unsqueeze(0).to(device)
                
                init_data_hist = hist['init_data']
                init_output = smpl.get_output(
                    global_orient=init_data_hist['init_global_orient'].to(device),
                    body_pose=init_data_hist['init_body_pose'].to(device),
                    betas=init_data_hist['init_betas'].to(device),
                    pose2rot=False,
                    return_full_pose=True
                )
                init_kp3d = root_centering(init_output.joints[:, :17], 'coco')
                init_kp = torch.cat((init_kp3d.reshape(1, -1), norm_kp2d[0, 0].clone().reshape(1, -1)), dim=-1).unsqueeze(0).to(device)
                
                import lib.utils.transforms as pt_transforms
                init_smpl = pt_transforms.matrix_to_rotation_6d(init_output.full_pose).unsqueeze(0).to(device)
                init_root = pt_transforms.matrix_to_rotation_6d(init_output.global_orient).to(device)
                
                cam_angvel = torch.zeros((1, norm_kp2d.shape[1], 6)).to(device)
                
                inits = (init_kp, init_smpl)

                try:
                    with torch.no_grad():
                        pred = network(
                            norm_kp2d,
                            inits,
                            feature_t,
                            mask=mask,
                            init_root=init_root,
                            cam_angvel=cam_angvel,
                            return_y_up=True,
                            cam_intrinsics=intrinsics.unsqueeze(0).to(device),
                            bbox=bbox_t.unsqueeze(0).to(device),
                            res=res.unsqueeze(0).to(device),
                            refine_traj=False,
                            states=None
                        )

                    verts_cam = extract_latest_vertices(pred).detach().cpu().numpy()
                    smplx_params = extract_latest_smplx_params(pred)
                    wham_Q.put((frame_id, start_time, frame, True, verts_cam, track_id, smplx_params))
                except Exception:
                    logger.exception(f"WHAM inference failed on frame {frame_id}")
                    wham_Q.put((frame_id, start_time, frame, False, None, track_id, None))
            else:
                wham_Q.put((frame_id, start_time, frame, False, None, None, None))

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
            if stream_npz_dir is not None:
                stream_chunk.append((int(frame_id), gvhmr_params, float(start_time)))
                flush_stream_chunk(force=False)
        
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
                verts_tensor = torch.from_numpy(verts_cam).to(device=device, dtype=torch.float32)
                # Render mesh expects bounding box update internally
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if renderer is not None:
                        rendered_frame = renderer.render_mesh(verts_tensor, frame_rgb, colors=[0.2, 0.4, 0.8])
                        frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
                except Exception:
                    render_error_count += 1
                    if render_error_count <= 5 or render_error_count % 50 == 0:
                        logger.exception(f"Render failed on frame {frame_id} (count={render_error_count})")
                cv2.putText(frame, f"Frame {frame_id} | FPS: {avg_fps:.1f} | Latency: {latency:.2f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame, f"Frame {frame_id} | FPS: {avg_fps:.1f} | Latency: {latency:.2f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            frame_rgb_out = frame[..., ::-1]
            if is_webcam:
                if writer is None:
                    writer_buffer.append(frame_rgb_out)
                    writer_buffer_ts.append(float(curr_time))
                    ensure_webcam_writer(force=False)
                else:
                    writer.append_data(frame_rgb_out)
            elif writer is not None:
                writer.append_data(frame_rgb_out)

            if is_webcam:
                cv2.imshow('WHAM Stream', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    logger.info("User requested exit from preview window.")
                    terminal_stop_event.set()
                    request_stop()
                    break

        if frame_id % 10 == 0:
            logger.info(f"Frame {frame_id} processed | FPS: {avg_fps:.1f} | Latency: {latency:.3f} s")
            
    flush_stream_chunk(force=True)
    if stream_npz_dir is not None:
        done_flag = os.path.join(stream_npz_dir, 'stream_done.flag')
        with open(done_flag, 'w') as f:
            # Write total chunk count so the consumer can wait for full consumption.
            f.write(str(int(stream_chunk_index)))

    if record_whamvideo and is_webcam:
        ensure_webcam_writer(force=True)
    if writer is not None:
        writer.close()

    for t, name in ((t1, 'reader'), (t2, 'detector'), (t3, 'extractor'), (t4, 'wham')):
        t.join(timeout=1.0)
        if t.is_alive():
            logger.warning(f"{name} thread did not exit in time; continue shutdown.")
    if terminal_listener is not None and terminal_listener.is_alive():
        stop_event.set()
        terminal_listener.join(timeout=0.5)

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
            if is_webcam and record_whamvideo:
                cv2.destroyAllWindows()
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
    if is_webcam and record_whamvideo:
        cv2.destroyAllWindows()
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
                        help='Optional directory to write incremental SMPL-X npz chunks for streaming.')
    parser.add_argument('--stream_chunk_size', type=int, default=16,
                        help='Number of frames per stream chunk npz.')
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
        stream_chunk_size=max(1, int(args.stream_chunk_size)),
        capture_time=max(0.0, float(args.time)),
    )
