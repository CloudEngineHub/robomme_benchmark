"""Initial wrist view (vertical for Video tasks, tilted forward for Button tasks) and the return home."""

import numpy as np

from robomme_plus.tasks.permanence.utils.reachability import CoverReachability


def camera_params(camera):
    p = camera.get_params()
    return (
        p["extrinsic_cv"].detach().cpu().numpy().reshape(3, 4),
        p["intrinsic_cv"].detach().cpu().numpy().reshape(3, 3),
    )


def inside_camera(points, extrinsic, intrinsic, margin=8):
    camera = np.c_[points, np.ones(len(points))] @ extrinsic.T
    pixels = camera @ intrinsic.T
    uv = pixels[:, :2] / np.maximum(pixels[:, 2:], 1e-12)
    return (
        (camera[:, 2] > 0)
        & (uv[:, 0] >= margin)
        & (uv[:, 0] < 256 - margin)
        & (uv[:, 1] >= margin)
        & (uv[:, 1] < 256 - margin)
    )


VIEW_POSITION = [0.0, 0.0, 0.20]

# Button tasks look forward to keep the area near the base out of view; Video tasks
# watch the demonstration with a vertical (top-down) wrist camera.
VIEW_TILT_DEGREES = {
    "VideoUnmask": 0.0,
    "VideoUnmaskSwap": 0.0,
    "ButtonUnmask": 30.0,
    "ButtonUnmaskSwap": 30.0,
}

NEAR_BASE_BOUNDS = np.array([[-0.49, -0.32], [-0.32, 0.32]])


def view_quaternion(tilt_degrees):
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", [180, -tilt_degrees, 0], degrees=True).as_quat()[
        [3, 0, 1, 2]
    ]


def initialize_view(native):
    """Set the initial robot configuration before any cue/execution step."""
    tilt = VIEW_TILT_DEGREES[native._limit_config["task"]]
    reach = CoverReachability(native)
    goal = reach.planner.transform_goal_to_wrt_base(
        np.r_[VIEW_POSITION, view_quaternion(tilt)]
    )
    status, solutions = reach.planner.IK(
        goal, reach.qpos, n_init_qpos=20, threshold=1e-4
    )
    if status != "Success":
        raise RuntimeError(f"Initial viewing pose is unreachable: {status}")
    qpos = np.asarray(solutions[0]).copy()
    qpos[-2:] = 0.04
    native.agent.reset(qpos)
    if native.gpu_sim_enabled:
        native.scene._gpu_apply_all()
        native.scene._gpu_fetch_all()
    native._forward_view_qpos = qpos
    native._limit_view_ready = True
    native._get_obs_sensor_data()
    ext, intrinsic = camera_params(native._sensors["hand_camera"])
    eye = -ext[:, :3].T @ ext[:, 3]
    xx, yy = np.meshgrid(
        np.linspace(*NEAR_BASE_BOUNDS[0], 65), np.linspace(*NEAR_BASE_BOUNDS[1], 65)
    )
    fractions = {}
    for z in [0.02, 0.11]:
        points = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, z)]
        fractions[str(z)] = float(
            inside_camera(points, ext, intrinsic, margin=-8).mean()
        )
    axis = ext[2, :3]
    if tilt:
        assert axis[0] > 0.4 and axis[2] < -0.5, axis
    else:
        assert axis[2] < -0.95, axis
    assert max(fractions.values()) == 0, fractions
    native._limit_diagnostics["shared_initial_view"] = dict(
        reference="central TCP position, "
        + ("tilted forward" if tilt else "vertical (top-down) wrist camera"),
        tcp_xyz_m=VIEW_POSITION,
        tcp_quaternion_wxyz=view_quaternion(tilt).tolist(),
        tilt_degrees=tilt,
        initial_qpos=qpos.tolist(),
        actual_tcp_pose=native.agent.tcp.pose.raw_pose.detach().cpu().numpy().tolist(),
        wrist_optical_axis_world=axis.tolist(),
        wrist_camera_eye_world=eye.tolist(),
        near_base_xy_bounds_m=NEAR_BASE_BOUNDS.tolist(),
        near_base_frustum_fraction_by_height=fractions,
        actual_wrist_extrinsic=ext.tolist(),
        actual_wrist_intrinsic=intrinsic.tolist(),
        simulator_steps_consumed=0,
        image_masks_used=False,
    )


def return_home_with_board(env, planner):
    import robomme_plus.control.reset as reset_panda

    native = env.unwrapped
    home = np.asarray(reset_panda.get_reset_panda_param("qpos"), dtype=float)
    start = native.agent.robot.get_qpos().detach().cpu().numpy().reshape(-1).copy()
    for step in range(20):
        blend = min((step + 1) / 12, 1)
        env.step(np.r_[start[:7] + blend * (home[:7] - start[:7]), 1.0])
    native._home_return_done = True
    native._limit_diagnostics["return_home_after_video"] = dict(
        control_steps=20,
        conditioning_history_excludes_home_motion=True,
        board_removed=False,
        board_stationary_during_return=True,
        execution_starts_at_home=True,
    )
