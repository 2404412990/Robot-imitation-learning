import argparse
import csv
import inspect
import os
import pickle
import pathlib
import sys
import time
import warnings
from collections import deque

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from rich import print

# Ensure repository root is on sys.path when launching via
# `python scripts/smplx_to_robot_stream.py`.
HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast


def _exp_smooth(curr, prev, alpha):
    if prev is None or alpha >= 0.999:
        return curr
    return alpha * curr + (1.0 - alpha) * prev


def _resolve_torch_device(requested):
    req = str(requested).strip().lower()
    if req == "cpu":
        return "cpu"
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda:0"
        print("[Stream] Warning: --torch_device=cuda but CUDA is unavailable, fallback to CPU.")
        return "cpu"
    # auto
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class OnlineQposPostprocessor:
    def __init__(
        self,
        xml_file,
        smooth_alpha=0.25,
        height_adjust=True,
        root_origin_offset=True,
        torch_device="auto",
    ):
        self.smooth_alpha = float(smooth_alpha)
        self.height_adjust = bool(height_adjust)
        self.root_origin_offset = bool(root_origin_offset)
        self.prev_qpos = None
        self.xy_origin = None

        device = _resolve_torch_device(torch_device)
        self.device = device
        self.kinematics_model = KinematicsModel(xml_file, device=device)

    def process(self, qpos):
        q = np.asarray(qpos, dtype=np.float32).copy()

        if self.prev_qpos is not None and self.smooth_alpha < 0.999:
            q[:3] = _exp_smooth(q[:3], self.prev_qpos[:3], self.smooth_alpha)

            quat = q[3:7].copy()
            prev_quat = self.prev_qpos[3:7]
            if np.dot(quat, prev_quat) < 0.0:
                quat = -quat
            quat = _exp_smooth(quat, prev_quat, self.smooth_alpha)
            quat /= (np.linalg.norm(quat) + 1e-8)
            q[3:7] = quat

            q[7:] = _exp_smooth(q[7:], self.prev_qpos[7:], self.smooth_alpha)

        if self.height_adjust:
            root_pos = torch.from_numpy(q[:3][None]).to(self.device, dtype=torch.float32)
            root_rot_xyzw = torch.from_numpy(q[3:7][[1, 2, 3, 0]][None]).to(self.device, dtype=torch.float32)
            dof_pos = torch.from_numpy(q[7:][None]).to(self.device, dtype=torch.float32)
            with torch.no_grad():
                body_pos, _ = self.kinematics_model.forward_kinematics(root_pos, root_rot_xyzw, dof_pos)
                lowest_height = torch.min(body_pos[..., 2]).item()
            q[2] -= lowest_height

        if self.root_origin_offset:
            if self.xy_origin is None:
                self.xy_origin = q[:2].copy()
            q[:2] -= self.xy_origin

        self.prev_qpos = q.copy()
        return q


def write_motion_pkl(save_path, qpos_seq, fps, xml_file, torch_device="auto"):
    qpos_seq = np.asarray(qpos_seq, dtype=np.float32)
    root_pos = qpos_seq[:, :3].copy()
    root_rot = qpos_seq[:, 3:7][:, [1, 2, 3, 0]].copy()  # wxyz -> xyzw
    dof_pos = qpos_seq[:, 7:].copy()

    device = _resolve_torch_device(torch_device)
    kinematics_model = KinematicsModel(xml_file, device=device)
    with torch.no_grad():
        fk_root_pos = torch.zeros((dof_pos.shape[0], 3), device=device)
        fk_root_rot = torch.zeros((dof_pos.shape[0], 4), device=device)
        fk_root_rot[:, -1] = 1.0
        local_body_pos, _ = kinematics_model.forward_kinematics(
            fk_root_pos,
            fk_root_rot,
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32),
        )
        local_body_pos = local_body_pos.detach().cpu().numpy()

    motion_data = {
        "fps": float(fps),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos,
        "link_body_list": kinematics_model.body_names,
    }

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    import pickle

    with open(save_path, "wb") as f:
        pickle.dump(motion_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_npz_dir", type=str, required=True, help="Directory containing stream artifacts (tail/chunk).")
    parser.add_argument(
        "--stream_mode",
        choices=["chunk", "tail", "hybrid"],
        default="tail",
        help="Stream transport mode. Default is tail (append-only stream_tail.pkl).",
    )
    parser.add_argument(
        "--stream_tail_path",
        type=str,
        default=None,
        help="Path to append-only tail stream file. Defaults to <stream_npz_dir>/stream_tail.pkl.",
    )
    parser.add_argument(
        "--torch_device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Torch device used by GMR postprocessing and FK routines.",
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "unitree_h1",
            "unitree_h1_2",
            "booster_t1",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "kuavo_s45",
            "hightorque_hi",
            "galaxea_r1pro",
            "berkeley_humanoid_lite",
            "booster_k1",
            "pnd_adam_lite",
            "openloong",
            "tienkung",
            "fourier_gr3",
        ],
        default="unitree_g1",
    )
    parser.add_argument(
        "--robot_path",
        type=str,
        default=None,
        help="Optional robot XML path. If set, overrides built-in XML for the selected --robot.",
    )
    parser.add_argument("--coord_fix", choices=["auto", "none", "yup_to_zup"], default="auto")
    parser.add_argument("--save_path", type=str, default="pkl_outputs/live_motion.pkl")
    parser.add_argument("--csv_path", type=str, default="pkl_outputs/csv/live_motion.csv")
    parser.add_argument("--record_gmrvideo", action="store_true")
    parser.add_argument("--record_video", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--video_path", type=str, default="videos/live_stream_robot.mp4")
    parser.add_argument(
        "--viewer_warmup_frames",
        type=int,
        default=0,
        help="Skip rendering first N frames in viewer while still writing csv/pkl.",
    )
    parser.add_argument(
        "--camera_follow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Follow robot base with viewer camera.",
    )
    parser.add_argument(
        "--camera_lookat_height_offset",
        type=float,
        default=0.45,
        help="Raise camera look-at point by this many meters.",
    )
    parser.add_argument(
        "--camera_elevation",
        type=float,
        default=12.0,
        help="Viewer camera elevation angle in degrees.",
    )
    parser.add_argument(
        "--camera_distance_scale",
        type=float,
        default=0.85,
        help="Scale factor for default robot camera distance.",
    )
    parser.add_argument(
        "--camera_azimuth",
        type=float,
        default=None,
        help="Optional fixed camera azimuth angle in degrees.",
    )

    parser.add_argument("--smooth_alpha", type=float, default=0.35)
    parser.add_argument("--height_adjust", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--root_origin_offset", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--poll_interval", type=float, default=0.05)
    parser.add_argument("--idle_timeout", type=float, default=0.0, help="Exit after idle timeout (seconds). 0 disables timeout.")
    parser.add_argument("--done_flag_name", type=str, default="stream_done.flag")
    parser.add_argument(
        "--ready_flag_path",
        type=str,
        default=None,
        help="Optional file path to touch when consumer startup is ready.",
    )
    parser.add_argument(
        "--viewer_startup_buffer_sec",
        type=float,
        default=0.8,
        help="Pre-buffer this many seconds before starting viewer playback.",
    )
    parser.add_argument(
        "--viewer_max_buffer_sec",
        type=float,
        default=1.5,
        help="Max buffered seconds for viewer; older frames are dropped when exceeded.",
    )
    parser.add_argument(
        "--viewer_drop_old_frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop oldest buffered viewer frames when lagging to keep playback responsive.",
    )
    parser.add_argument(
        "--done_grace_sec",
        type=float,
        default=2.0,
        help="Extra wait time after done flag appears before exiting, to avoid missing late chunk files.",
    )
    parser.add_argument(
        "--max_chunk_retry",
        type=int,
        default=120,
        help="Maximum retries per chunk before exiting with an error to avoid infinite loops.",
    )

    args = parser.parse_args()
    stream_mode = str(args.stream_mode).strip().lower()
    if stream_mode not in ("chunk", "tail", "hybrid"):
        print(f"[Stream] Warning: unknown stream_mode={stream_mode}, fallback to tail.")
        stream_mode = "tail"
    if stream_mode == "hybrid":
        print("[Stream] Warning: hybrid mode is not supported in consumer yet, fallback to tail.")
        stream_mode = "tail"

    record_gmrvideo = bool(args.record_gmrvideo or args.record_video)
    robot_path = args.robot_path if args.robot_path is not None and str(args.robot_path).strip() != "" else None

    gmr_init_params = set(inspect.signature(GMR.__init__).parameters.keys())
    viewer_init_params = set(inspect.signature(RobotMotionViewer.__init__).parameters.keys())
    gmr_supports_robot_path = "robot_path" in gmr_init_params

    if robot_path is not None and not gmr_supports_robot_path:
        print(
            "[Stream] Fatal: current GeneralMotionRetargeting does not support --robot_path. "
            "Please update the code/environment or unset ROBOT_PATH."
        )
        sys.exit(2)

    stream_npz_dir = os.path.abspath(args.stream_npz_dir)
    done_flag = os.path.join(stream_npz_dir, args.done_flag_name)
    stream_tail_path = os.path.abspath(args.stream_tail_path) if args.stream_tail_path else os.path.join(stream_npz_dir, "stream_tail.pkl")
    os.makedirs(stream_npz_dir, exist_ok=True)

    csv_dir = os.path.dirname(args.csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    csv_file = open(args.csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    processed = set()
    qpos_history = []
    aligned_fps_holder = [30.0]

    state = {
        "retarget": None,
        "postprocessor": None,
        "viewer": None,
        "tail_body_model": None,
        "tail_joint_names": None,
        "tail_parents": None,
    }
    viewer_init_attempted = [False]
    tail_coord_fix_logged = [False]
    viewer_queue = deque()
    viewer_started = [False]
    viewer_drop_count = [0]

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

    def tail_params_to_smplx_frame(params):
        if state["tail_body_model"] is None:
            body_model = smplx.create(
                str(SMPLX_FOLDER),
                "smplx",
                gender="neutral",
                use_pca=False,
            )
            state["tail_body_model"] = body_model
            state["tail_joint_names"] = JOINT_NAMES[: len(body_model.parents)]
            state["tail_parents"] = body_model.parents

        body_model = state["tail_body_model"]
        joint_names = state["tail_joint_names"]
        parents = state["tail_parents"]

        body_pose = np.asarray(params["body_pose"], dtype=np.float32).reshape(1, 63)
        root_orient = np.asarray(params["global_orient_global"], dtype=np.float32).reshape(1, 3)
        trans = np.asarray(params["transl_global"], dtype=np.float32).reshape(1, 3)

        if args.coord_fix in ("auto", "yup_to_zup"):
            root_orient, trans = _tail_apply_yup_to_zup(root_orient, trans)
            if not tail_coord_fix_logged[0]:
                print("[SMPL] Applied coordinate fix: y-up -> z-up")
                tail_coord_fix_logged[0] = True

        betas = _align_tail_betas(params["betas"], body_model)

        with torch.no_grad():
            smplx_output = body_model(
                betas=torch.from_numpy(betas).float().view(1, -1),
                global_orient=torch.from_numpy(root_orient).float(),
                body_pose=torch.from_numpy(body_pose).float(),
                transl=torch.from_numpy(trans).float(),
                left_hand_pose=torch.zeros(1, 45).float(),
                right_hand_pose=torch.zeros(1, 45).float(),
                jaw_pose=torch.zeros(1, 3).float(),
                leye_pose=torch.zeros(1, 3).float(),
                reye_pose=torch.zeros(1, 3).float(),
                return_full_pose=True,
            )

        single_global_orient = smplx_output.global_orient[0].detach().cpu().numpy()
        single_full_body_pose = smplx_output.full_pose[0].detach().cpu().numpy().reshape(-1, 3)
        single_joints = smplx_output.joints[0].detach().cpu().numpy()

        result = {}
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(single_full_body_pose[i].squeeze())
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))

        return result, betas

    def init_viewer_if_needed(motion_fps):
        if not record_gmrvideo:
            return
        if state["viewer"] is not None:
            return
        if viewer_init_attempted[0]:
            return

        viewer_init_attempted[0] = True
        viewer_kwargs = dict(
            robot_type=args.robot,
            motion_fps=float(max(1.0, motion_fps)),
            transparent_robot=0,
            record_video=True,
            video_path=args.video_path,
        )
        if robot_path is not None and "robot_path" in viewer_init_params:
            viewer_kwargs["robot_path"] = robot_path
        if "camera_follow" in viewer_init_params:
            viewer_kwargs["camera_follow"] = args.camera_follow
        if "camera_lookat_height_offset" in viewer_init_params:
            viewer_kwargs["camera_lookat_height_offset"] = args.camera_lookat_height_offset
        if "camera_elevation" in viewer_init_params:
            viewer_kwargs["camera_elevation"] = args.camera_elevation
        if "camera_distance_scale" in viewer_init_params:
            viewer_kwargs["camera_distance_scale"] = args.camera_distance_scale
        if "camera_azimuth" in viewer_init_params:
            viewer_kwargs["camera_azimuth"] = args.camera_azimuth
        try:
            state["viewer"] = RobotMotionViewer(**viewer_kwargs)
            print(f"[Stream] MuJoCo viewer enabled (DISPLAY={os.environ.get('DISPLAY', '<unset>')})")
        except Exception as e:
            state["viewer"] = None
            print(
                "[Stream] Warning: failed to initialize MuJoCo viewer "
                f"(DISPLAY={os.environ.get('DISPLAY', '<unset>')}): {e}. "
                "Continue without on-screen window. "
                "If you expect a window: ensure DISPLAY is valid, run `xhost +local:docker` for Docker, "
                "and avoid USE_XVFB_GMR=1."
            )

    def init_retarget_if_needed(actual_human_height=None):
        if state["retarget"] is not None:
            return

        retarget_kwargs = dict(
            actual_human_height=actual_human_height,
            src_human="smplx",
            tgt_robot=args.robot,
        )
        if robot_path is not None and gmr_supports_robot_path:
            retarget_kwargs["robot_path"] = robot_path
        state["retarget"] = GMR(**retarget_kwargs)

        state["postprocessor"] = OnlineQposPostprocessor(
            state["retarget"].xml_file,
            smooth_alpha=max(0.0, min(1.0, args.smooth_alpha)),
            height_adjust=args.height_adjust,
            root_origin_offset=args.root_origin_offset,
            torch_device=args.torch_device,
        )

        init_viewer_if_needed(aligned_fps_holder[0])

    def _viewer_target_fps():
        return float(max(1.0, aligned_fps_holder[0]))

    def _viewer_required_frames():
        return max(0, int(round(max(0.0, args.viewer_startup_buffer_sec) * _viewer_target_fps())))

    def _ensure_viewer_started(force=False):
        if state["viewer"] is None:
            return False
        if viewer_started[0]:
            return True

        qlen = len(viewer_queue)
        required = _viewer_required_frames()
        if qlen >= required or (force and qlen > 0):
            viewer_started[0] = True
            print(
                f"[Stream] Viewer playback starts (buffer={qlen}, required={required}, "
                f"drop_old={int(bool(args.viewer_drop_old_frames))})"
            )
        return viewer_started[0]

    def play_one_viewer_frame(force_start=False):
        if state["viewer"] is None:
            return False
        if not _ensure_viewer_started(force=force_start):
            return False
        if len(viewer_queue) == 0:
            return False

        qpos = viewer_queue.popleft()
        state["viewer"].step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=state["retarget"].scaled_human_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=True,
            follow_camera=args.camera_follow,
        )
        return True

    def enqueue_viewer_qpos(qpos):
        if state["viewer"] is None:
            return
        viewer_queue.append(np.asarray(qpos, dtype=np.float32).copy())
        if args.viewer_drop_old_frames:
            max_frames = max(1, int(round(max(0.1, args.viewer_max_buffer_sec) * _viewer_target_fps())))
            overflow = len(viewer_queue) - max_frames
            if overflow > 0:
                for _ in range(overflow):
                    viewer_queue.popleft()
                viewer_drop_count[0] += int(overflow)

    def process_chunk(chunk_path):
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            chunk_path, SMPLX_FOLDER, coord_fix=args.coord_fix
        )
        smplx_frames, aligned_fps_local = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=30
        )
        aligned_fps_holder[0] = float(aligned_fps_local)

        if state["retarget"] is None:
            init_retarget_if_needed(actual_human_height=actual_human_height)

        for frame in smplx_frames:
            qpos = state["retarget"].retarget(frame)
            qpos = state["postprocessor"].process(qpos)
            qpos_history.append(qpos.copy())

            # CSV uses root_rot in xyzw order.
            row = np.concatenate([qpos[:3], qpos[3:7][[1, 2, 3, 0]], qpos[7:]], axis=0)
            csv_writer.writerow(row.tolist())

            if state["viewer"] is not None and (len(qpos_history) - 1) >= max(0, int(args.viewer_warmup_frames)):
                enqueue_viewer_qpos(qpos)
                play_one_viewer_frame(force_start=False)

        csv_file.flush()

    def process_tail_record(record):
        frame_id = int(record.get("frame_id", -1))
        params = record.get("params")
        if frame_id < 0 or params is None:
            return None

        smplx_frame, betas = tail_params_to_smplx_frame(params)

        if state["retarget"] is None:
            human_height = None
            if betas is not None and betas.size > 0:
                human_height = 1.66 + 0.1 * float(betas[0])
            init_retarget_if_needed(actual_human_height=human_height)

        qpos = state["retarget"].retarget(smplx_frame)
        qpos = state["postprocessor"].process(qpos)
        qpos_history.append(qpos.copy())

        row = np.concatenate([qpos[:3], qpos[3:7][[1, 2, 3, 0]], qpos[7:]], axis=0)
        csv_writer.writerow(row.tolist())

        if state["viewer"] is not None and (len(qpos_history) - 1) >= max(0, int(args.viewer_warmup_frames)):
            enqueue_viewer_qpos(qpos)
            play_one_viewer_frame(force_start=False)

        csv_file.flush()
        return frame_id

    print(f"[Stream] Watching {stream_npz_dir}")
    print(f"[Stream] Mode: {stream_mode}")
    if stream_mode in ("tail", "hybrid"):
        print(f"[Stream] Tail path: {stream_tail_path}")
    print(f"[Stream] CSV output: {args.csv_path}")

    # Initialize MuJoCo viewer before signaling readiness so upstream can
    # wait until the GMR visualization path is actually ready.
    init_viewer_if_needed(aligned_fps_holder[0])

    if args.ready_flag_path:
        try:
            ready_flag_path = os.path.abspath(args.ready_flag_path)
            ready_dir = os.path.dirname(ready_flag_path)
            if ready_dir:
                os.makedirs(ready_dir, exist_ok=True)
            with open(ready_flag_path, "w") as f:
                f.write(f"pid={os.getpid()} ts={time.time()}\n")
            print(f"[Stream] Ready flag created: {ready_flag_path}")
        except Exception as e:
            print(f"[Stream] Warning: failed to create ready flag {args.ready_flag_path}: {e}")

    last_activity = time.time()
    fatal_error = None
    done_seen_at = None
    done_expected_chunks = None
    chunk_retry_count = {}
    tail_file = None
    tail_done_seen = False
    while True:
        progressed = False

        if stream_mode == "chunk":
            chunk_names = sorted(
                [n for n in os.listdir(stream_npz_dir) if n.startswith("chunk_") and n.endswith(".npz")]
            )
            new_chunk_names = [n for n in chunk_names if n not in processed]

            if len(new_chunk_names) > 0:
                done_seen_at = None
                for name in new_chunk_names:
                    chunk_path = os.path.join(stream_npz_dir, name)
                    try:
                        process_chunk(chunk_path)
                    except Exception as e:
                        retry = int(chunk_retry_count.get(name, 0)) + 1
                        chunk_retry_count[name] = retry

                        err_msg = str(e)
                        if "unexpected keyword argument 'robot_path'" in err_msg:
                            fatal_error = (
                                "GeneralMotionRetargeting in current runtime does not accept robot_path. "
                                "Please update environment or unset ROBOT_PATH."
                            )
                            print(f"[Stream] Fatal on {name}: {fatal_error}")
                            break

                        if retry <= 3 or retry % 20 == 0:
                            print(f"[Stream] Waiting for valid chunk {name} (retry={retry}): {e}")

                        if retry >= max(1, int(args.max_chunk_retry)):
                            fatal_error = f"Chunk {name} failed {retry} times. Last error: {e}"
                            print(f"[Stream] Fatal on {name}: {fatal_error}")
                            break
                        continue

                    processed.add(name)
                    chunk_retry_count.pop(name, None)
                    print(f"[Stream] Processed {name} (total chunks: {len(processed)})")
                    last_activity = time.time()
                    progressed = True

                if fatal_error is not None:
                    break

            if not progressed:
                if os.path.exists(done_flag):
                    if done_expected_chunks is None:
                        try:
                            with open(done_flag, "r") as f:
                                done_expected_chunks = int((f.read() or "").strip())
                                print(f"[Stream] Done flag detected with expected chunks={done_expected_chunks}")
                        except Exception:
                            done_expected_chunks = None

                    if done_expected_chunks is not None and len(processed) >= done_expected_chunks:
                        final_chunk_names = sorted(
                            [n for n in os.listdir(stream_npz_dir) if n.startswith("chunk_") and n.endswith(".npz")]
                        )
                        final_new_chunks = [n for n in final_chunk_names if n not in processed]
                        if len(final_new_chunks) == 0:
                            break

                    if done_expected_chunks is not None:
                        time.sleep(max(0.01, args.poll_interval))
                        continue

                    if done_seen_at is None:
                        done_seen_at = time.time()
                    elif (time.time() - done_seen_at) >= max(0.0, args.done_grace_sec):
                        final_chunk_names = sorted(
                            [n for n in os.listdir(stream_npz_dir) if n.startswith("chunk_") and n.endswith(".npz")]
                        )
                        final_new_chunks = [n for n in final_chunk_names if n not in processed]
                        if len(final_new_chunks) == 0:
                            break
                else:
                    done_seen_at = None

        else:  # tail mode
            if tail_file is None:
                if os.path.exists(stream_tail_path):
                    try:
                        tail_file = open(stream_tail_path, "rb")
                    except Exception as e:
                        fatal_error = f"Failed to open tail stream {stream_tail_path}: {e}"
                        print(f"[Stream] Fatal: {fatal_error}")
                        break
                else:
                    if args.idle_timeout > 0 and (time.time() - last_activity) > args.idle_timeout:
                        print(f"[Stream] Idle timeout ({args.idle_timeout}s), exiting.")
                        break
                    time.sleep(max(0.01, args.poll_interval))
                    continue

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=DeprecationWarning,
                        message=r"numpy\.core\.numeric is deprecated.*",
                    )
                    record = pickle.load(tail_file)
            except EOFError:
                if tail_done_seen:
                    break
            except Exception as e:
                fatal_error = f"Tail stream read failed: {e}"
                print(f"[Stream] Fatal: {fatal_error}")
                break
            else:
                if isinstance(record, dict) and bool(record.get("done", False)):
                    tail_done_seen = True
                elif isinstance(record, dict):
                    try:
                        frame_id = process_tail_record(record)
                    except Exception as e:
                        fatal_error = f"Tail record process failed: {e}"
                        print(f"[Stream] Fatal: {fatal_error}")
                        break
                    if frame_id is not None:
                        if frame_id not in processed:
                            processed.add(frame_id)
                            if len(processed) % 20 == 0:
                                print(f"[Stream] Processed tail frames: {len(processed)}")
                        last_activity = time.time()
                        progressed = True

        if fatal_error is not None:
            break

        if not progressed:
            if args.idle_timeout > 0 and (time.time() - last_activity) > args.idle_timeout:
                print(f"[Stream] Idle timeout ({args.idle_timeout}s), exiting.")
                break
            time.sleep(max(0.01, args.poll_interval))

    if tail_file is not None:
        tail_file.close()

    csv_file.flush()
    csv_file.close()

    if stream_mode == "chunk":
        print(
            f"[Stream] Summary: processed_chunks={len(processed)} "
            f"expected_chunks={done_expected_chunks if done_expected_chunks is not None else -1} "
            f"qpos_frames={len(qpos_history)}"
        )
    else:
        print(f"[Stream] Summary: processed_tail_frames={len(processed)} qpos_frames={len(qpos_history)}")

    if state["viewer"] is not None:
        drain_timeout = max(2.0, float(args.viewer_max_buffer_sec) + float(args.done_grace_sec) + 2.0)
        drain_deadline = time.time() + drain_timeout
        while len(viewer_queue) > 0 and time.time() < drain_deadline:
            played = play_one_viewer_frame(force_start=True)
            if not played:
                time.sleep(0.001)
        if len(viewer_queue) > 0:
            print(
                "[Stream] Warning: viewer queue did not fully drain before timeout; "
                f"remaining={len(viewer_queue)}"
            )
        print(f"[Stream] Viewer playback stopped. dropped_frames={viewer_drop_count[0]}")

    if len(qpos_history) > 0 and args.save_path:
        write_motion_pkl(
            args.save_path,
            qpos_history,
            aligned_fps_holder[0],
            state["retarget"].xml_file,
            torch_device=args.torch_device,
        )
        print(f"[Stream] Saved motion pkl: {args.save_path}")
    else:
        print("[Stream] No qpos generated; skip pkl save.")

    if state["viewer"] is not None:
        state["viewer"].close()

    if fatal_error is not None:
        print(f"[Stream] Exit with error: {fatal_error}")
        sys.exit(2)

    print("[Stream] Done.")
