import argparse
import pathlib
import os
import time

import numpy as np
import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel

from rich import print


def _exp_smooth(x, alpha):
    if x.shape[0] <= 1 or alpha >= 0.999:
        return x
    y = x.copy()
    for i in range(1, y.shape[0]):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def postprocess_qpos(
    qpos_seq,
    xml_file,
    smooth_alpha=0.35,
    height_adjust=True,
    root_origin_offset=True,
):
    qpos_seq = np.asarray(qpos_seq, dtype=np.float32)
    if qpos_seq.ndim != 2 or qpos_seq.shape[1] < 8:
        return qpos_seq

    qpos = qpos_seq.copy()

    # Smooth root translation, root quaternion, and all DoFs.
    if smooth_alpha < 0.999:
        qpos[:, :3] = _exp_smooth(qpos[:, :3], smooth_alpha)

        root_rot = qpos[:, 3:7].copy()
        # Keep quaternion hemisphere continuous before smoothing.
        for i in range(1, root_rot.shape[0]):
            if np.dot(root_rot[i], root_rot[i - 1]) < 0.0:
                root_rot[i] = -root_rot[i]
        root_rot = _exp_smooth(root_rot, smooth_alpha)
        root_rot /= (np.linalg.norm(root_rot, axis=1, keepdims=True) + 1e-8)
        qpos[:, 3:7] = root_rot

        qpos[:, 7:] = _exp_smooth(qpos[:, 7:], smooth_alpha)

    if not height_adjust and not root_origin_offset:
        return qpos

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    kinematics_model = KinematicsModel(xml_file, device=device)

    root_pos = qpos[:, :3].copy()
    root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]].copy()
    dof_pos = qpos[:, 7:].copy()

    if height_adjust:
        with torch.no_grad():
            body_pos, _ = kinematics_model.forward_kinematics(
                torch.from_numpy(root_pos).to(device=device, dtype=torch.float32),
                torch.from_numpy(root_rot_xyzw).to(device=device, dtype=torch.float32),
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32),
            )
            lowest_height = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] -= lowest_height

    if root_origin_offset:
        root_pos[:, :2] -= root_pos[0, :2]

    qpos[:, :3] = root_pos
    return qpos

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/TWIST-dev/motion_data/AMASS/KIT_572_dance_chacha11_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1Running_c3d/Run_C24_-_quick_side_step_left_stageii.npz",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung", "fourier_gr3"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--coord_fix",
        choices=["auto", "none", "yup_to_zup"],
        default="auto",
        help="Coordinate fix for input SMPL-X. Use 'yup_to_zup' for WHAM y-up outputs.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--smooth_alpha",
        type=float,
        default=0.35,
        help="Exponential smoothing alpha in [0,1]. Larger means less smoothing. 1 disables smoothing.",
    )

    parser.add_argument(
        "--height_adjust",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adjust root height so the lowest robot point touches ground.",
    )

    parser.add_argument(
        "--root_origin_offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offset root XY so first frame starts at origin.",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER, coord_fix=args.coord_fix
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",
    )

    # Retarget all frames first so we can apply stable post-processing.
    qpos_seq = []
    for smplx_data_frame in smplx_data_frames:
        qpos = retarget.retarget(smplx_data_frame)
        qpos_seq.append(qpos.copy())

    qpos_seq = np.asarray(qpos_seq, dtype=np.float32)
    qpos_seq = postprocess_qpos(
        qpos_seq,
        xml_file=retarget.xml_file,
        smooth_alpha=max(0.0, min(1.0, args.smooth_alpha)),
        height_adjust=args.height_adjust,
        root_origin_offset=args.root_origin_offset,
    )

    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
    
    # Start the viewer
    i = 0
    qpos_len = len(qpos_seq)

    while True:
        if args.loop:
            i = (i + 1) % qpos_len
        else:
            i += 1
            if i >= qpos_len:
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        qpos = qpos_seq[i]

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=smplx_data_frames[i],
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )
            
    if args.save_path is not None:
        import pickle
        root_pos = qpos_seq[:, :3].copy()
        # save from wxyz to xyzw
        root_rot = qpos_seq[:, 3:7][:, [1, 2, 3, 0]].copy()
        dof_pos = qpos_seq[:, 7:].copy()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        kinematics_model = KinematicsModel(retarget.xml_file, device=device)
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
        body_names = kinematics_model.body_names
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
            
      
    
    robot_motion_viewer.close()
