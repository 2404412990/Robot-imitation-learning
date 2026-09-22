import re
import sys

with open("handle_wham_gmr.py", "r") as f:
    content = f.read()

# 1. Add args parser
argp_orig = """    parser.add_argument('--time', type=float, default=0.0,
                        help='When --video=0 (webcam): >0 auto-stop after this many seconds; 0 waits for terminal q/ESC.')
    args = parser.parse_args()"""

argp_new = """    parser.add_argument('--time', type=float, default=0.0,
                        help='When --video=0 (webcam): >0 auto-stop after this many seconds; 0 waits for terminal q/ESC.')
    
    # GMR Args
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--robot_path", type=str, default=None)
    parser.add_argument("--torch_device", default="auto")
    parser.add_argument("--coord_fix", default="yup_to_zup")
    parser.add_argument("--save_path", type=str, default="pkl_outputs/live_motion.pkl")
    parser.add_argument("--csv_path", type=str, default="pkl_outputs/csv/live_motion.csv")
    parser.add_argument("--record_gmrvideo", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/live_stream_robot.mp4", help="GMR video path")
    parser.add_argument("--viewer_warmup_frames", type=int, default=12)
    parser.add_argument("--camera_follow", action="store_true")
    parser.add_argument("--no-camera_follow", dest="camera_follow", action="store_false")
    parser.set_defaults(camera_follow=False)
    parser.add_argument("--camera_lookat_height_offset", type=float, default=0.45)
    parser.add_argument("--camera_elevation", type=float, default=12.0)
    parser.add_argument("--camera_distance_scale", type=float, default=0.85)
    parser.add_argument("--camera_azimuth", type=float, default=None)
    parser.add_argument("--smooth_alpha", type=float, default=0.35)
    parser.add_argument("--height_adjust", action="store_true")
    parser.add_argument("--no-height_adjust", dest="height_adjust", action="store_false")
    parser.set_defaults(height_adjust=True)
    parser.add_argument("--root_origin_offset", action="store_true")
    parser.add_argument("--no-root_origin_offset", dest="root_origin_offset", action="store_false")
    parser.set_defaults(root_origin_offset=False)
    parser.add_argument("--poll_interval", type=float, default=0.05)
    parser.add_argument("--idle_timeout", type=float, default=0.0)
    parser.add_argument("--done_grace_sec", type=float, default=2.0)
    parser.add_argument("--viewer_ready_timeout_sec", type=float, default=8.0)
    parser.add_argument("--viewer_thread_join_timeout_sec", type=float, default=10.0)
    
    args = parser.parse_args()"""
content = content.replace(argp_orig, argp_new)

# 2. Modify run_stream_mt parameters
run_orig = """    record_whamvideo=False,
    stream_npz_dir=None,
    capture_time=0.0,
):"""

run_new = """    record_whamvideo=False,
    capture_time=0.0,
    gmr_args=None,
):"""
content = content.replace(run_orig, run_new)

# 3. Add imports at the top
import_orig = """import time
import os
import cv2"""

import_new = """import time
import os
import cv2
import csv
import smplx
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

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
"""
content = content.replace(import_orig, import_new)

# 4. Integrate GMR setup and processing
gmr_proc_orig = """    if tail_writer is not None:
        tail_writer.close(frame_count=len(frame_param_map))"""

queue_process_orig = """        if gvhmr_params is not None:
            frame_param_map[frame_id] = gvhmr_params
            frame_timestamp_map[frame_id] = float(start_time)
            if tail_writer is not None:
                tail_writer.write_record(frame_id, start_time, gvhmr_params)"""

gmr_setup_inj = """    # GMR State Setup
    gmr_state = {
        "retarget": None,
        "postprocessor": None,
        "viewer": None,
        "tail_body_model": None,
        "tail_joint_names": None,
        "tail_parents": None,
    }
    qpos_history = []
    csv_file = None
    csv_writer = None
    
    if gmr_args is not None:
        csv_dir = os.path.dirname(gmr_args.csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        csv_file = open(gmr_args.csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        
    def tail_params_to_smplx_frame(params):
        if gmr_state["tail_body_model"] is None:
            bm = smplx.create(str(SMPLX_FOLDER), "smplx", gender="neutral", use_pca=False)
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
                betas=torch.from_numpy(betas).float().view(1, -1),
                global_orient=torch.from_numpy(rorient).float(),
                body_pose=torch.from_numpy(bpose).float(),
                transl=torch.from_numpy(trans).float(),
                left_hand_pose=torch.zeros(1, 45).float(),
                right_hand_pose=torch.zeros(1, 45).float(),
                jaw_pose=torch.zeros(1, 3).float(),
                leye_pose=torch.zeros(1, 3).float(),
                reye_pose=torch.zeros(1, 3).float(),
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
            res[jn] = (sj[i], r.as_quat())

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
        gmr_state["postprocessor"] = OnlineQposPostprocessor(
            gmr_state["retarget"].xml_file,
            smooth_alpha=max(0.0, min(1.0, gmr_args.smooth_alpha)),
            height_adjust=gmr_args.height_adjust,
            root_origin_offset=gmr_args.root_origin_offset,
            torch_device=gmr_args.torch_device,
        )
        
        if gmr_args.record_gmrvideo:
            v_kws = dict(
                robot_type=gmr_args.robot,
                motion_fps=30.0,
                transparent_robot=0,
                record_video=True,
                video_path=gmr_args.video_path,
            )
            v_kws["camera_follow"] = getattr(gmr_args, "camera_follow", False)
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
                
            try:
                gmr_state["viewer"] = RobotMotionViewer(**v_kws)
                logger.info(f"Initialized GMR viewer for {gmr_args.robot}")
            except Exception as e:
                logger.error(f"Failed to initialize GMR viewer: {e}")
                gmr_state["viewer"] = None
"""
# inject gmr state setup before frame_times = deque
content = content.replace("    frame_times = deque(maxlen=30)", gmr_setup_inj + "\n    frame_times = deque(maxlen=30)")

# replace queue process
new_queue_process = """        if gvhmr_params is not None:
            frame_param_map[frame_id] = gvhmr_params
            frame_timestamp_map[frame_id] = float(start_time)
            
            # Process via GMR immediately in queue thread
            if gmr_args is not None:
                sf, bt = tail_params_to_smplx_frame(gvhmr_params)
                init_gmr(bt)
                
                qp = gmr_state["retarget"].retarget(sf)
                qp = gmr_state["postprocessor"].process(qp)
                qpos_history.append(qp.copy())
                
                row = np.concatenate([qp[:3], qp[3:7][[1, 2, 3, 0]], qp[7:]], axis=0)
                csv_writer.writerow(row.tolist())
                csv_file.flush()
                
                if gmr_state["viewer"] is not None and (len(qpos_history) - 1) >= max(0, int(gmr_args.viewer_warmup_frames)):
                    try:
                        gmr_state["viewer"].step(
                            root_pos=qp[:3],
                            root_rot=qp[3:7],
                            dof_pos=qp[7:],
                            human_motion_data=None,
                            human_pos_offset=np.array([0.0, 0.0, 0.0]),
                            show_human_body_name=False,
                            rate_limit=False,
                            follow_camera=gmr_args.camera_follow,
                        )
                    except Exception as e:
                        logger.error(f"GMR viewer error: {e}")"""
content = content.replace(queue_process_orig, new_queue_process)
                        
cleanup_orig = """            logger.info(f"Frame {frame_id} processed | FPS: {avg_fps:.1f} | Latency: {latency:.3f} s")
            
    if tail_writer is not None:"""
    
cleanup_new = """            logger.info(f"Frame {frame_id} processed | FPS: {avg_fps:.1f} | Latency: {latency:.3f} s")
            
    if gmr_args is not None and csv_file is not None:
        csv_file.close()
        from scripts.smplx_to_robot_stream import write_motion_pkl
        if len(qpos_history) > 0:
            logger.info(f"Saving motion PKL to {gmr_args.save_path}")
            try:
                write_motion_pkl(gmr_args.save_path, qpos_history, fps=30.0, xml_file=gmr_state["retarget"].xml_file, torch_device=gmr_args.torch_device)
            except Exception as e:
                logger.error(f"Failed to write motion pkl: {e}")
        if gmr_state["viewer"] is not None:
            try:
                gmr_state["viewer"].close()
            except:
                pass"""
content = content.replace(cleanup_orig, cleanup_new)

# Remove stream_npz_dir parsing logic and replace run_stream_mt call
call_orig = """    run_stream_mt(
        cfg,
        args.video,
        args.output_dir,
        rotate_deg=args.rotate,
        record_whamvideo=record_whamvideo,
        stream_npz_dir=args.stream_npz_dir,
        capture_time=max(0.0, float(args.time)),
    )"""

call_new = """    run_stream_mt(
        cfg,
        args.video,
        args.output_dir,
        rotate_deg=args.rotate,
        record_whamvideo=record_whamvideo,
        capture_time=max(0.0, float(args.time)),
        gmr_args=args,
    )"""
content = content.replace(call_orig, call_new)

with open("handle_wham_gmr.py", "w") as f:
    f.write(content)
