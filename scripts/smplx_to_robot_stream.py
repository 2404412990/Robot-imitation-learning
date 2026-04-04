import argparse
import csv
import inspect
import os
import pathlib
import sys
import time

import numpy as np
import torch

from rich import print

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast


def _exp_smooth(curr, prev, alpha):
    if prev is None or alpha >= 0.999:
        return curr
    return alpha * curr + (1.0 - alpha) * prev


class OnlineQposPostprocessor:
    def __init__(self, xml_file, smooth_alpha=0.25, height_adjust=True, root_origin_offset=True):
        self.smooth_alpha = float(smooth_alpha)
        self.height_adjust = bool(height_adjust)
        self.root_origin_offset = bool(root_origin_offset)
        self.prev_qpos = None
        self.xy_origin = None

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
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


def write_motion_pkl(save_path, qpos_seq, fps, xml_file):
    qpos_seq = np.asarray(qpos_seq, dtype=np.float32)
    root_pos = qpos_seq[:, :3].copy()
    root_rot = qpos_seq[:, 3:7][:, [1, 2, 3, 0]].copy()  # wxyz -> xyzw
    dof_pos = qpos_seq[:, 7:].copy()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_npz_dir", type=str, required=True, help="Directory containing chunk_*.npz stream files.")
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
    parser.add_argument("--rate_limit", action="store_true")
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
        default=0.75,
        help="Raise camera look-at point by this many meters.",
    )
    parser.add_argument(
        "--camera_elevation",
        type=float,
        default=-5.0,
        help="Viewer camera elevation angle in degrees.",
    )
    parser.add_argument(
        "--camera_distance_scale",
        type=float,
        default=1.0,
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
    parser.add_argument("--root_origin_offset", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--poll_interval", type=float, default=0.05)
    parser.add_argument("--idle_timeout", type=float, default=0.0, help="Exit after idle timeout (seconds). 0 disables timeout.")
    parser.add_argument("--done_flag_name", type=str, default="stream_done.flag")
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
    }

    def process_chunk(chunk_path):
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            chunk_path, SMPLX_FOLDER, coord_fix=args.coord_fix
        )
        smplx_frames, aligned_fps_local = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=30
        )
        aligned_fps_holder[0] = float(aligned_fps_local)

        if state["retarget"] is None:
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
            )
            if record_gmrvideo:
                viewer_kwargs = dict(
                    robot_type=args.robot,
                    motion_fps=aligned_fps_holder[0],
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
                state["viewer"] = RobotMotionViewer(**viewer_kwargs)

        for frame in smplx_frames:
            qpos = state["retarget"].retarget(frame)
            qpos = state["postprocessor"].process(qpos)
            qpos_history.append(qpos.copy())

            # CSV uses root_rot in xyzw order.
            row = np.concatenate([qpos[:3], qpos[3:7][[1, 2, 3, 0]], qpos[7:]], axis=0)
            csv_writer.writerow(row.tolist())

            if state["viewer"] is not None and (len(qpos_history) - 1) >= max(0, int(args.viewer_warmup_frames)):
                state["viewer"].step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=state["retarget"].scaled_human_data,
                    human_pos_offset=np.array([0.0, 0.0, 0.0]),
                    show_human_body_name=False,
                    rate_limit=args.rate_limit,
                    follow_camera=args.camera_follow,
                )

        csv_file.flush()

    print(f"[Stream] Watching {stream_npz_dir}")
    print(f"[Stream] CSV output: {args.csv_path}")

    last_activity = time.time()
    done_seen_at = None
    done_expected_chunks = None
    chunk_retry_count = {}
    fatal_error = None
    while True:
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

                    # Avoid log flooding for transient chunk write races.
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

            if fatal_error is not None:
                break
            continue

        if fatal_error is not None:
            break

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
                # Final rescan before exit to catch chunks that landed right around done_flag creation.
                final_chunk_names = sorted(
                    [n for n in os.listdir(stream_npz_dir) if n.startswith("chunk_") and n.endswith(".npz")]
                )
                final_new_chunks = [n for n in final_chunk_names if n not in processed]
                if len(final_new_chunks) == 0:
                    break
            time.sleep(max(0.01, args.poll_interval))
            continue
        else:
            done_seen_at = None

        if args.idle_timeout > 0 and (time.time() - last_activity) > args.idle_timeout:
            print(f"[Stream] Idle timeout ({args.idle_timeout}s), exiting.")
            break

        time.sleep(max(0.01, args.poll_interval))

    csv_file.flush()
    csv_file.close()

    print(
        f"[Stream] Summary: processed_chunks={len(processed)} "
        f"expected_chunks={done_expected_chunks if done_expected_chunks is not None else -1} "
        f"qpos_frames={len(qpos_history)}"
    )

    if len(qpos_history) > 0 and args.save_path:
        write_motion_pkl(args.save_path, qpos_history, aligned_fps_holder[0], state["retarget"].xml_file)
        print(f"[Stream] Saved motion pkl: {args.save_path}")
    else:
        print("[Stream] No qpos generated; skip pkl save.")

    if state["viewer"] is not None:
        state["viewer"].close()

    if fatal_error is not None:
        print(f"[Stream] Exit with error: {fatal_error}")
        sys.exit(2)

    print("[Stream] Done.")
