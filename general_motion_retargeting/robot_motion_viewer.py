import os
import subprocess
import time
from contextlib import nullcontext
import mujoco as mj
import mujoco.viewer as mjv
import imageio
import glfw
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

class RobotMotionViewer:
    def __init__(
        self,
        robot_type,
        camera_follow=True,
        motion_fps=30,
        transparent_robot=0,
        # video recording
        record_video=False,
        video_path=None,
        video_width=640,
        video_height=480,
        window_width=None,
        window_height=None,
        keyboard_callback=None,
        robot_path=None,
        camera_lookat_height_offset=0.75,
        camera_elevation=-5.0,
        camera_distance_scale=1.0,
        camera_azimuth=None,
    ):

        self.robot_type = robot_type
        self.xml_path = robot_path if robot_path is not None and str(robot_path).strip() != "" else ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)

        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.camera_lookat_height_offset = float(camera_lookat_height_offset)
        self.camera_elevation = float(camera_elevation)
        self.camera_distance_scale = float(camera_distance_scale)
        self.camera_azimuth = camera_azimuth
        self.record_video = record_video
        self.window_width = window_width
        self.window_height = window_height

        # --- Fix HiDPI + fullscreen issues on Linux ---
        # On Wayland / X11, tell GLFW to respect the requested window size
        # rather than auto-scaling to the monitor's content scale.
        for _k in ("GDK_SCALE", "GDK_DPI_SCALE", "QT_AUTO_SCREEN_SCALE_FACTOR",
                    "QT_SCALE_FACTOR", "ELM_SCALE"):
            os.environ.setdefault(_k, "1")
        os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")

        # MuJoCoʼs render_loop may call glfwDefaultWindowHints() internally,
        # which would wipe all our hints.  Neutralise it so our settings stick.
        _orig_default_hints = glfw.default_window_hints
        def _patched_default_hints():
            print("[viewer] glfw.default_window_hints() called — blocked to preserve our hints")
        glfw.default_window_hints = _patched_default_hints

        glfw.init()
        glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.FALSE)
        glfw.window_hint(glfw.MAXIMIZED, glfw.FALSE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        glfw.window_hint(glfw.FOCUSED, glfw.TRUE)
        glfw.window_hint(glfw.FLOATING, glfw.FALSE)
        glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)

        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=keyboard_callback
            )

        glfw.default_window_hints = _orig_default_hints

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot

        # Best-effort window resize via X11 tools (no-op on pure Wayland).
        target_window_width = int(window_width) if window_width is not None else 1200
        target_window_height = int(window_height) if window_height is not None else 900
        if not self._set_glfw_window_size(self.viewer, target_window_width, target_window_height):
            self._fix_window_size(target_window_width, target_window_height)
        
        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
            
            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)
        
    @staticmethod
    def _set_glfw_window_size(viewer_handle, width, height):
        """Resize the MuJoCo viewer when the GLFW window handle is reachable."""
        candidates = [
            viewer_handle,
            getattr(viewer_handle, "_viewer", None),
            getattr(viewer_handle, "_sim", None),
        ]

        for candidate in candidates:
            if candidate is None:
                continue
            for attr in ("_window", "window"):
                window = getattr(candidate, attr, None)
                if window is None:
                    continue
                try:
                    glfw.set_window_size(window, int(width), int(height))
                    return True
                except Exception:
                    continue

        return False

    @staticmethod
    def _fix_window_size(width, height):
        """Try to resize the MuJoCo viewer window to *width* x *height*.

        On HiDPI Linux the initial window can be much larger than the
        requested dimensions because of display-server scaling.  This
        helper attempts to shrink it with X11 command-line tools (best
        effort — failures are silent).
        """
        time.sleep(0.2)  # let the window manager settle
        for tool, args in (
            ("xdotool", [
                "search", "--sync", "--name", "MuJoCo",
                "windowsize", "--usehints", str(width), str(height),
            ]),
            ("wmctrl", [
                "-r", "MuJoCo", "-e",
                "0,-1,-1,{w},{h}".format(w=width, h=height),
            ]),
        ):
            try:
                subprocess.run(
                    [tool] + args,
                    timeout=2,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                break  # first successful tool wins
            except Exception:
                continue

    def step(self,
            # robot data
            root_pos, root_rot, dof_pos,
            # human data
            human_motion_data=None,
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True,
            follow_camera=True,
            ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """

        root_pos = np.asarray(root_pos, dtype=np.float64)
        root_rot = np.asarray(root_rot, dtype=np.float64)
        dof_pos = np.asarray(dof_pos, dtype=np.float64)
        expected_dofs = int(self.model.nq) - 7
        if root_pos.shape[0] != 3 or root_rot.shape[0] != 4 or dof_pos.shape[0] != expected_dofs:
            raise ValueError(
                f"Viewer qpos shape mismatch for {self.robot_type}: "
                f"root_pos={root_pos.shape[0]}, root_rot={root_rot.shape[0]}, "
                f"dof_pos={dof_pos.shape[0]}, expected_dof={expected_dofs}, model_nq={self.model.nq}"
            )

        lock_context = self.viewer.lock() if hasattr(self.viewer, "lock") else nullcontext()
        with lock_context:
            self.data.qpos[:3] = root_pos
            self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
            self.data.qpos[7:] = dof_pos

            mj.mj_forward(self.model, self.data)

            if follow_camera:
                try:
                    lookat = self.data.xpos[self.model.body(self.robot_base).id].copy()
                except Exception:
                    lookat = np.asarray(self.data.qpos[:3], dtype=np.float64).copy()
                lookat[2] += self.camera_lookat_height_offset
                self.viewer.cam.lookat[:] = lookat
                self.viewer.cam.distance = self.viewer_cam_distance * max(0.1, self.camera_distance_scale)
                self.viewer.cam.elevation = self.camera_elevation
                if self.camera_azimuth is not None:
                    self.viewer.cam.azimuth = float(self.camera_azimuth)

            if human_motion_data is not None:
                # Clean custom geometry
                self.viewer.user_scn.ngeom = 0
                # Draw the task targets for reference
                for human_body_name, (pos, rot) in human_motion_data.items():
                    draw_frame(
                        pos,
                        R.from_quat(np.asarray(rot)[[1, 2, 3, 0]]).as_matrix(),  # wxyz -> xyzw
                        self.viewer,
                        human_point_scale,
                        pos_offset=human_pos_offset,
                        joint_name=human_body_name if show_human_body_name else None
                        )

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use renderer for proper offscreen rendering
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)
    
    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
