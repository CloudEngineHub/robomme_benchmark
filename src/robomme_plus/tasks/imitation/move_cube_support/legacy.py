"""Replay-only implementation for frozen pre-five-method MoveCube records.

New episodes use ..move_cube.MoveCube. Historical oracle budgets and scene
records must not silently acquire different geometry or success criteria.
"""

import copy
from typing import Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS

# Robomme
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from robomme_plus.simulation.base import MemoryTask
from robomme_plus.simulation.difficulty import (
    normalize_robomme_difficulty,
    require_xhard4_only,
)
from robomme_plus.simulation.specification import EpisodeSpecError, SpecRecorder
from robomme_plus.common.errors import SceneGenerationError
from robomme_plus.simulation.actors import (
    point_segment_distance_xy,
    spawn_random_cube,
    spawn_random_target,
)
from robomme_plus.common.errors import SamplingConfigError


# Must match the robot base pose in MoveCube._load_agent.
ROBOT_BASE_XY = (-0.615, 0.0)
# Push-start offsets; must match solve_push_to_target / solve_push_to_target_with_peg.
PUSH_BACKOFF_M = 0.10
PEG_PUSH_LATERAL_M = 0.10


SCENE_CONFIG = {'parameters': {'peg_size': {'length_expression': '0.1 + (0.05 - 0.05) * rand()',
                             'radius_expression': '0.01 + (0.005 - 0.005) * rand()',
                             'note': 'Both rand results are zeroed by the 0 factor but must be '
                                     'kept so the random stream is not shifted (R8)'},
                'way_selection': {'sampler': 'torch.randint(len(self.ways))'},
                'obj_selection': {'sampler': 'torch.randint(0, 2)', 'mapping': [-1, 1]},
                'dir_sample': {'sampler': 'torch.randint(0, 2)',
                               'consumed': False,
                               'note': 'drawn but unused; kept for sampling_trace'},
                'direction_rule': 'evaluate gives +/-1 from the actual y difference between the '
                                  'two layouts; no longer drawn randomly',
                'reset_rule': 'step switches to the already generated execution pose',
                'recovery': 'this environment has no inject_fail_grasp; it only accepts the entry '
                            'recovery mode, and the actual recovery action is null'},
 'positions': {'goal_demo': {'region_center': [0.0, 0.0],
                             'region_half_size': 0.15,
                             'radius_factor': 2,
                             'min_gap_factor': 1},
               'goal_execution': {'region_center': [0.0, 0.0],
                                  'region_half_size': 0.1,
                                  'radius_factor': 2,
                                  'min_gap_factor': 1},
               'cube_rejection': {'max_trials': 128, 'min_distance_factor': 5},
               'peg_color': {'head': '#EC7357', 'tail': '#EC7357'}},
 'decision': {'demo_layout': {'peg_position_policy': {'base_y_abs': 0.2,
                                                      'base_y_threshold': 0.5,
                                                      'jitter_span': 0.1},
                              'cube_position_policy': {'center_span': 0.2,
                                                       'center_offset': -0.1,
                                                       'region_half_size': 0.05},
                              'xhard4': {'region': {'center': [-0.06, 0.0],
                                                    'r_in': 0.24,
                                                    'r_out': 0.42,
                                                    'base_dist': [0.31, 0.8],
                                                    'push_len_max': 0.3,
                                                    'peg_gap': 0.04,
                                                    'goal_peg_gap': 0.02,
                                                    'peg_max_trials': 128,
                                                    'goal_max_trials': 256,
                                                    'cube_max_trials': 4096}}},
              'execution_layout': {'peg_position_policy': {'base_y_abs': 0.2,
                                                           'base_y_threshold': 0.5,
                                                           'jitter_span': 0.1},
                                   'cube_position_policy': {'center_span': 0.2,
                                                            'center_offset': -0.1,
                                                            'region_half_size': 0.05},
                                   'xhard4': {'region': {'center': [-0.06, 0.0],
                                                         'r_in': 0.24,
                                                         'r_out': 0.42,
                                                         'base_dist': [0.31, 0.8],
                                                         'push_len_max': 0.3,
                                                         'peg_gap': 0.04,
                                                         'goal_peg_gap': 0.02,
                                                         'peg_max_trials': 128,
                                                         'goal_max_trials': 256,
                                                         'cube_max_trials': 4096}}},
              'peg_yaw_range': {'span_rad': 1.5707963267948966,
                                'offset_rad': 0.7853981633974483,
                                'xhard4': {'span_rad': 6.283185307179586,
                                           'offset_rad': 3.141592653589793}}}}


def _peg_axis_extent(length):
    """Interval ``(t_min, t_max)`` (meters) of the peg axis segment along peg direction u in the peg-root frame.

    Derived from the geometry in ``utils/object_generation.py::build_peg``: the head link is centered on the
    peg root and the tail link hangs at ``-length*u`` via a fixed joint; collision boxes have half-length
    ``0.45*length`` and visual boxes ``0.5*length``. The union of both (i.e. the visual extent) is used:
    ``(-0.15, +0.05)`` for ``length=0.1``, matching the measured value.
    After building each xhard4 peg, ``MoveCube._xhard4_verify_peg_extent`` re-checks it against the actual
    shapes and raises on a mismatch.
    """
    head_half = 0.5 * float(length)
    tail_center = -float(length)
    return (tail_center - head_half, head_half)


def _peg_root_xy(base_y, x_jitter, y_jitter):
    """Peg root xy (float64), computed bit for bit like the float32 translation used when ``_load_scene`` builds the peg."""
    translation = np.array([0.0, base_y, 0.0], dtype=np.float32)
    translation[1] = base_y
    translation[:2] += np.array([x_jitter, y_jitter], dtype=np.float32)
    return translation[:2].astype(np.float64)


def _peg_geometry(root, yaw, length, extent):
    """Peg root ``root`` and heading ``yaw`` -> ``(grasp point xy, peg segment endpoints a, b)`` (float64).

    Grasp point = tail link center = ``root - length*u`` (``evaluate`` sets ``grasp_target = peg_tail``, which
    ``grasp_and_lift_peg_side`` grasps); peg segment = ``root + t*u`` for ``t`` in ``extent``.
    """
    u = np.array([np.cos(float(yaw)), np.sin(float(yaw))], dtype=np.float64)
    root = np.asarray(root, dtype=np.float64)
    return root - float(length) * u, root + extent[0] * u, root + extent[1] * u


def _in_region_u(xy, region):
    """Whether a point (object center or grasp point) is inside unified region U: annulus and base-distance band. Returns a description string on violation, else None."""
    xy = np.asarray(xy, dtype=np.float64)
    rc = float(np.linalg.norm(xy - region["center"]))
    if not (region["r_in"] <= rc <= region["r_out"]):
        return f"distance to annulus center {rc:.6f} not in [{region['r_in']}, {region['r_out']}]"
    rb = float(np.linalg.norm(xy - np.asarray(ROBOT_BASE_XY, dtype=np.float64)))
    if not (region["base_lo"] <= rb <= region["base_hi"]):
        return f"distance to base {rb:.6f} not in [{region['base_lo']}, {region['base_hi']}]"
    return None


def _peg_region_violation(root, yaw, length, extent, region):
    """Peg rule: the grasp point is inside U and the peg segment stays at least r_in from the annulus center (the whole peg avoids the inner hole). Returns a description string on violation."""
    grasp, a, b = _peg_geometry(root, yaw, length, extent)
    why = _in_region_u(grasp, region)
    if why is not None:
        return f"grasp point: {why}"
    d = point_segment_distance_xy(region["center"], a, b)
    if d < region["r_in"]:
        return f"peg segment distance to annulus center {d:.6f} < r_in {region['r_in']}"
    return None


def _assert_peg_in_region(root, yaw, length, extent, region, spec_prefix):
    """N17: replayed frozen specs skip the rejection loop, so re-check the peg pose with the same rule; raises ``EpisodeSpecError`` on violation."""
    why = _peg_region_violation(root, yaw, length, extent, region)
    if why is not None:
        raise EpisodeSpecError(
            f"MoveCube xhard4: {spec_prefix}.peg_offsets/peg_yaw violate unified region U (frozen or injected values): {why}"
        )


class _TaskScene(MemoryTask):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[Panda]
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)
    _clearance = 0.01


    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0,
        seed=0,
        Robomme_video_episode=None,
        Robomme_video_path=None,
        sampling_config=None,
        native_episode_spec=None,
        **kwargs,
    ):
        # Must run before any random number call and super().__init__()
        self._sampling = copy.deepcopy(SCENE_CONFIG)
        self._spec = SpecRecorder(
            native_episode_spec,
            "MoveCube",
            {"seed": seed},
            difficulty=kwargs.get("difficulty"),
        )
        self._native_init_index = -1
        self.reset_in_proecess = False
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]

        self.seed = seed
        self._hb_generator = torch.Generator()
        self._hb_generator.manual_seed(int(self.seed))

        self.robomme_failure_recovery = bool(
            kwargs.pop("robomme_failure_recovery", False)
        )
        self.robomme_failure_recovery_mode = kwargs.pop(
            "robomme_failure_recovery_mode", None
        )
        if isinstance(self.robomme_failure_recovery_mode, str):
            self.robomme_failure_recovery_mode = (
                self.robomme_failure_recovery_mode.lower()
            )
        normalized_robomme_difficulty = normalize_robomme_difficulty(
            kwargs.pop("difficulty", None)
        )
        if normalized_robomme_difficulty is not None:
            self.difficulty = normalized_robomme_difficulty
        else:
            seed_mod = seed % 3
            if seed_mod == 0:
                self.difficulty = "easy"
            elif seed_mod == 1:
                self.difficulty = "medium"
            else:
                self.difficulty = "hard"
        require_xhard4_only(self.difficulty, "MoveCube")
        self._xhard_peg_yaw_reduction = True

        self.restore_flag = False
        self.use_demonstrationwrapper = False
        self.demonstration_record_traj = False
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()

        length_tensor = torch.rand(1, generator=self._hb_generator)
        radius_tensor = torch.rand(1, generator=self._hb_generator)
        self.length = (0.1 + (0.05 - 0.05) * length_tensor).item()
        self.radius = (0.01 + (0.005 - 0.005) * radius_tensor).item()

        # Create a single peg
        # peg_spawn_translation = np.array([self.length / 2, 0.0, self.radius], dtype=np.float32)
        demo_layout = self._sampling["decision"]["demo_layout"]
        exec_layout = self._sampling["decision"]["execution_layout"]
        peg_yaw_range = self._sampling["decision"]["peg_yaw_range"]
        native_pos = self._sampling["positions"]
        # xhard4 uses its own branch; the original path below must stay unchanged.
        self._load_scene_xhard4_region(
            demo_layout, exec_layout, peg_yaw_range, native_pos
        )
        return

    def _sample_obj_and_dir(self):
        """The obj_sample and dir_sample draws (shared by the original tiers and xhard4; same order as the original code)."""
        obj_sample = self._spec.value(
            "objects.obj_sample",
            int(torch.randint(0, 2, (1,), generator=self._hb_generator).item()),
        )
        self.obj_flag = -1 if obj_sample == 0 else 1
        # Drawn but unused; the draw must still happen to keep the random stream unchanged.
        dir_sample = self._spec.value(
            "objects.sampling_trace.dir_sample",
            int(torch.randint(0, 2, (1,), generator=self._hb_generator).item()),
        )
        # self.direction = -1 if dir_sample.item() == 0 else 1

    def _store_goal_poses(self):
        """float64 copies of both segments' goal disc poses (draws no random numbers; shared by the original tiers and xhard4)."""
        goal2_p = np.array(
            self.goal_site_2.pose.p.detach().cpu().numpy(), dtype=np.float64, copy=True
        )
        self.goal_site_2_pose_p = goal2_p

        goal2_q = np.array(
            self.goal_site_2.pose.q.detach().cpu().numpy(), dtype=np.float64, copy=True
        )
        self.goal_site_2_pose_q = goal2_q

        goal1_p = np.array(
            self.goal_site.pose.p.detach().cpu().numpy(), dtype=np.float64, copy=True
        )
        self.goal_site_1_pose_p = goal1_p

        goal1_q = np.array(
            self.goal_site.pose.q.detach().cpu().numpy(), dtype=np.float64, copy=True
        )
        self.goal_site_1_pose_q = goal1_q

    def _load_scene_xhard4_region(
        self, demo_layout, exec_layout, peg_yaw_range, native_pos
    ):
        """V6 xhard4 layout: cube center, goal center and peg grasp point all share unified region U (annulus and base-distance band).

        Random call order (this branch only, re-frozen for xhard4):
        demo-segment peg (3 ``torch.rand`` per trial: grasp x, y, yaw; a rejection redraws the whole group) ->
        execution-segment peg (same) -> obj_sample, dir_sample -> demo-segment goal -> execution-segment goal ->
        demo-segment cube -> execution-segment cube.
        V5's base_y sampling, two-level cube candidate-center sampling, ``center_exclusion`` disc exclusion and
        separate demo/execution goal boxes are all removed.

        Peg: grasp point (peg tail = peg root - length*u) uniform in U, yaw in +/-pi, peg root derived from both;
        the peg segment stays at least r_in from the annulus center.
        Goal: center in U, at least goal_peg_gap from this segment's peg.
        Cube: center in U, at least peg_gap from this segment's peg, distance to this segment's goal in
        [min_cg, push_len_max], and all three push start points inside the base-distance band.
        Spec keys keep the original path: ``layout.<seg>.peg_offsets = [0.0, peg root x, peg root y]``
        (base_y is always 0; the two "jitter" values are the peg root xy).
        Replayed frozen values skip the rejection loop; the peg is re-checked here and goal/cube by the spawn
        functions under the same rule (N17).
        """
        yaw_policy = peg_yaw_range["xhard4"]
        dk_yaw = "peg_yaw_range.xhard4"
        peg_extent = _peg_axis_extent(self.length)
        regions = {
            "demo": self._xhard4_region(demo_layout, "demo_layout"),
            "execution": self._xhard4_region(exec_layout, "execution_layout"),
        }
        trials = {"demo": {}, "execution": {}}
        pegs = {}
        for seg, label in (
            ("demo", "demo segment"),
            ("execution", "execution segment"),
        ):
            region = regions[seg]
            root_x, root_y, yaw, trials[seg]["peg_trials"] = (
                self._xhard4_sample_peg_in_region(region, yaw_policy, peg_extent, label)
            )
            base_y, root_x, root_y = self._spec.value(
                f"layout.{seg}.peg_offsets",
                [0.0, root_x, root_y],
                decision_key=f"{region['key']}.xhard4.region",
            )
            yaw = self._spec.value(f"layout.{seg}.peg_yaw", yaw, decision_key=dk_yaw)
            root = _peg_root_xy(base_y, root_x, root_y)
            # Replay returns frozen values from the two calls above without the rejection loop; re-check with the same rule
            _assert_peg_in_region(
                root, yaw, self.length, peg_extent, region, f"layout.{seg}"
            )
            translation = np.array([0.0, base_y, 0.0], dtype=np.float32)
            translation[1] = base_y
            translation[:2] += np.array([root_x, root_y], dtype=np.float32)
            yaw_angles = torch.tensor([[0.0, 0.0, yaw]], dtype=torch.float32)
            yaw_quat = (
                matrix_to_quaternion(
                    euler_angles_to_matrix(yaw_angles, convention="XYZ")
                )[0]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            pose = sapien.Pose(p=translation.tolist(), q=yaw_quat)
            _, seg_a, seg_b = _peg_geometry(root, yaw, self.length, peg_extent)
            pegs[seg] = {
                "translation": translation,
                "pose": pose,
                "segment": (seg_a, seg_b),
            }
            if seg == "demo":
                self.peg1_basex = translation[0]
                self.peg1_basey = translation[1]
                self.peg, self.peg_head, self.peg_tail = self._build_imitation_tool(
                    length=self.length,
                    radius=self.radius,
                    initial_pose=pose,
                    name="peg",
                    head_color="#EC7357",
                    tail_color="#EC7357",
                )
                # The peg axis segment used by the peg rule must match the built peg's actual geometry
                self._xhard4_verify_peg_extent(peg_extent)
                self.pegs = [self.peg]
                self.peg_heads = [self.peg_head]
                self.peg_tails = [self.peg_tail]
                self.peg_init_poses = [peg.pose for peg in self.pegs]
                self.peg_init_pose = self.pegs[0].pose
            else:
                self.peg2_basex = translation[0]
                self.peg2_basey = translation[1]
                self.peg_init_poses_2 = [pose]
        self.finish_return_flag = False

        self._sample_obj_and_dir()

        base_xy = tuple(ROBOT_BASE_XY)
        goals = {}
        for seg, name, attr, goal_key, label in (
            ("demo", "goal_site", "goal_site", "goal_demo", "demo segment"),
            (
                "execution",
                "goal_site_2",
                "goal_site_2",
                "goal_execution",
                "execution segment",
            ),
        ):
            region = regions[seg]
            radius = self.cube_half_size * native_pos[goal_key]["radius_factor"]
            try:
                goal = spawn_random_target(
                    self,
                    avoid=None,
                    include_existing=False,
                    include_goal=False,
                    region_center=[
                        float(region["center"][0]),
                        float(region["center"][1]),
                    ],
                    region_half_size=region["r_out"] + radius,
                    radius=radius,
                    thickness=0.005,
                    min_gap=self.cube_half_size * 1,
                    name_prefix=name,
                    max_trials=region["goal_max_trials"],
                    recorder=self._spec,
                    spec_path=f"layout.{seg}.goal_xy",
                    generator=self._hb_generator,
                    annulus=region["annulus"],
                    base_band=(base_xy, region["base_lo"], region["base_hi"]),
                    segment_clearance=[(*pegs[seg]["segment"], region["goal_peg_gap"])],
                )
            except RuntimeError as exc:
                if not isinstance(exc, SceneGenerationError):
                    raise SceneGenerationError(
                        f"MoveCube xhard4: {label} goal generation failed: {exc}"
                    ) from exc
                raise
            setattr(self, attr, goal)
            goals[seg] = np.asarray(goal.pose.p, dtype=np.float64).reshape(-1)[:2]

        min_cg = (
            self.cube_half_size * native_pos["cube_rejection"]["min_distance_factor"]
        )
        for seg, name, attr, label in (
            ("demo", "fixed_cube", "cube", "demo segment"),
            ("execution", "fixed_cube_2", "cube_2", "execution segment"),
        ):
            region = regions[seg]
            try:
                cube = spawn_random_cube(
                    self,
                    region_center=[
                        float(region["center"][0]),
                        float(region["center"][1]),
                    ],
                    color=(1, 0, 0, 1),
                    name_prefix=name,
                    recorder=self._spec,
                    spec_path=f"layout.{seg}.cube_pose",
                    region_half_size=region["r_out"] + self.cube_half_size,
                    generator=self._hb_generator,
                    half_size=self.cube_half_size,
                    max_trials=region["cube_max_trials"],
                    include_existing=False,
                    include_goal=False,
                    annulus=region["annulus"],
                    base_band=(base_xy, region["base_lo"], region["base_hi"]),
                    segment_clearance=[(*pegs[seg]["segment"], region["peg_gap"])],
                    push_feasible=(
                        tuple(goals[seg]),
                        min_cg,
                        region["push_len_max"],
                        PUSH_BACKOFF_M,
                        PEG_PUSH_LATERAL_M,
                        base_xy,
                        region["base_lo"],
                        region["base_hi"],
                    ),
                )
            except RuntimeError as exc:
                if not isinstance(exc, SceneGenerationError):
                    raise SceneGenerationError(
                        f"MoveCube xhard4: {label} cube generation failed: {exc}"
                    ) from exc
                raise
            setattr(self, attr, cube)
        self.cube_init_pose = self.cube.pose
        self.cube_init_pose_2 = self.cube_2.pose
        self._store_goal_poses()

        # Read-only records; keep after all sampling points (draws no random numbers)
        for seg in ("demo", "execution"):
            self._spec.record(
                f"layout.{seg}.region",
                dict(
                    regions[seg]["decision"],
                    peg_axis_extent_m=list(peg_extent),
                    robot_base_xy=list(ROBOT_BASE_XY),
                    push_backoff_m=PUSH_BACKOFF_M,
                    peg_push_lateral_m=PEG_PUSH_LATERAL_M,
                    min_cube_goal_m=min_cg,
                ),
            )
            self._spec.record(f"layout.{seg}.region_trials", dict(trials[seg]))

    def _xhard4_region(self, layout, key):
        """Fetch and validate xhard4's unified region U; reject missing fields or invalid values instead of silently relaxing them."""
        cfg = layout["xhard4"]["region"]
        where = f"MoveCube xhard4: decision.{key}.xhard4.region"
        if not isinstance(cfg, dict):
            raise SamplingConfigError(f"{where} must be a dict, got {cfg!r}")
        try:
            center = np.asarray(cfg["center"], dtype=np.float64).reshape(-1)
            r_in, r_out = float(cfg["r_in"]), float(cfg["r_out"])
            base_lo, base_hi = (float(v) for v in cfg["base_dist"])
            nums = {
                k: float(cfg[k]) for k in ("push_len_max", "peg_gap", "goal_peg_gap")
            }
            budgets = {
                k: cfg[k]
                for k in ("peg_max_trials", "goal_max_trials", "cube_max_trials")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise SamplingConfigError(
                f"{where} missing field or wrong type: {exc}"
            ) from exc
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise SamplingConfigError(
                f"{where}.center must be two finite numbers, got {cfg['center']!r}"
            )
        if not (np.isfinite(r_in) and np.isfinite(r_out) and 0.0 <= r_in < r_out):
            raise SamplingConfigError(
                f"{where} must satisfy 0 <= r_in < r_out, got ({cfg['r_in']!r}, {cfg['r_out']!r})"
            )
        if not (
            np.isfinite(base_lo) and np.isfinite(base_hi) and 0.0 <= base_lo < base_hi
        ):
            raise SamplingConfigError(
                f"{where}.base_dist must be [lo, hi] with 0 <= lo < hi, got {cfg['base_dist']!r}"
            )
        for k, v in nums.items():
            if not (np.isfinite(v) and v >= 0.0):
                raise SamplingConfigError(
                    f"{where}.{k} must be a finite number >= 0, got {cfg[k]!r}"
                )
        for k, v in budgets.items():
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, np.integer))
                or int(v) < 1
            ):
                raise SamplingConfigError(
                    f"{where}.{k} must be an integer >= 1, got {v!r}"
                )
        return {
            "key": key,
            "center": center,
            "r_in": r_in,
            "r_out": r_out,
            "annulus": ((float(center[0]), float(center[1])), r_in, r_out),
            "base_lo": base_lo,
            "base_hi": base_hi,
            **nums,
            **{k: int(v) for k, v in budgets.items()},
            "decision": copy.deepcopy(cfg),
        }

    def _xhard4_sample_peg_in_region(self, region, yaw_policy, extent, seg_label):
        """V6 xhard4: each trial draws one ``torch.rand`` each for (grasp x, grasp y, yaw), derives the peg root and
        checks the peg rule; a violation redraws the whole group. The grasp point is uniform in the square
        circumscribing the annulus (uniform in U after rejection).

        Returns ``(peg root x, peg root y, yaw, trial count)``. The peg root that is checked goes through the same
        float32 translation as peg construction (``_peg_root_xy``), so replay re-checks and generation judge
        identically bit for bit. Raises a genuine ``SceneGenerationError`` after ``peg_max_trials`` trials.
        """
        cx, cy = float(region["center"][0]), float(region["center"][1])
        side = 2.0 * region["r_out"]
        for trial in range(1, region["peg_max_trials"] + 1):
            gx = torch.rand(1, generator=self._hb_generator).item() * side + (
                cx - region["r_out"]
            )
            gy = torch.rand(1, generator=self._hb_generator).item() * side + (
                cy - region["r_out"]
            )
            yaw = (
                torch.rand(1, generator=self._hb_generator).item()
                * (yaw_policy["span_rad"])
                - (yaw_policy["offset_rad"])
            )
            root_x = gx + self.length * float(np.cos(yaw))
            root_y = gy + self.length * float(np.sin(yaw))
            root = _peg_root_xy(0.0, root_x, root_y)
            if _peg_region_violation(root, yaw, self.length, extent, region) is None:
                return float(root_x), float(root_y), float(yaw), trial
        raise SceneGenerationError(
            f"MoveCube xhard4: {seg_label} peg violated unified region U on all {region['peg_max_trials']} resamples"
        )


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._native_init_index = getattr(self, "_native_init_index", -1) + 1
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "pegs"):
                return

            # Initialize all 3 pegs

            qpos = np.array(
                [
                    0.0,
                    0,
                    0,
                    -np.pi * 4 / 8,
                    0,
                    np.pi * 2 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ],
                dtype=np.float32,
            )
            self.ways = ["peg_push", "gripper_push", "grasp_putdown"]
            way_idx = self._spec.value(
                f"initializations.{getattr(self, '_native_init_index', 0)}.way_idx",
                torch.randint(
                    len(self.ways), (1,), generator=self._hb_generator
                ).item(),
            )
            self.way = self.ways[way_idx]
            # self.way="gripper_push"

            self.agent.reset(qpos)
            self._peg_grasp_flipped = False
            self._peg_grasp_flip_log = []
            self.cube_2.set_pose(sapien.Pose(p=[10, 10, 1]))  # only need the pose!
            self.goal_site_2.set_pose(sapien.Pose(p=[10, -10, 1]))


    # Robomme


from robomme_plus.tasks.imitation.utils.environment import ImitationMixin


class MoveCube(ImitationMixin, _TaskScene):
    """Repeat the demonstrated tool use (hook, poke, or scoop) in a new layout."""

    task_name = "MoveCube"


from robomme_plus.tasks.imitation.utils.tools import setup_move

setup_move(MoveCube)
