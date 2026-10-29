"""Panda stick robot whose stick is cut short in execution (PatternLock, RouteStick).

The stick of ManiSkill's ``panda_stick`` is one 10 cm cylinder under the hand,
with the TCP at its end. This robot splits it at the episode's sampled fraction
into a fixed upper part and a lower part on its own link, and adds
``stick_short_tip`` at the end of the upper part. When execution starts:

- the lower part is hidden and stops colliding, leaving 50-70% in PatternLock
  v6 medium, 35-60% in v6 hard, 40-60% in v5, or 25-50% in older versions
  and RouteStick;
- the arm keeps its pose, so the shortened tip is higher than the full
  tip was: the robot has to move down to draw;
- scoring uses the real (short) tip.

The TCP (observations, waypoint commands, motion planning) stays at the
full-length position, beyond the visible tip by the removed length.

The optional tilted preview swaps the straight full-length cylinder for a
short cylinder on a rotated mount when execution starts. Its marker follows
the physical stick axis; the gripper must compensate for this mount rotation.
"""

import hashlib
import re
import tempfile
from pathlib import Path

import numpy as np

UID = "panda_split_stick"
STICK_START_M, STICK_LENGTH_M = (
    0.05,
    0.10,
)  # the native stick spans 5-15 cm below the hand frame
LOWER_LINK, SHORT_TIP_LINK = "stick_lower", "stick_short_tip"
CHANGED_LINK = 'stick_changed'
_FULL = """    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.008" length="0.1"/>
      </geometry>
    </visual>"""
_FULL_COLLISION = _FULL.replace("<visual>", "<collision>").replace(
    "</visual>", "</collision>"
)


def _cylinder(tag, center, length):
    return f"""    <{tag}>
      <origin xyz="0 0 {center:.9f}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.008" length="{length:.9f}"/>
      </geometry>
    </{tag}>"""


def _lower_links(fraction):
    kept = STICK_LENGTH_M * fraction
    removed = STICK_LENGTH_M - kept
    split = STICK_START_M + kept
    center = (split + STICK_START_M + STICK_LENGTH_M) / 2
    return f"""
  <link name="{LOWER_LINK}">
{_cylinder("visual", center, removed)}
{_cylinder("collision", center, removed)}
  </link>
  <joint name="{LOWER_LINK}_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="panda_hand"/>
    <child link="{LOWER_LINK}"/>
  </joint>
  <link name="{SHORT_TIP_LINK}"/>
  <joint name="{SHORT_TIP_LINK}_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 {split:.9f}"/>
    <parent link="panda_hand"/>
    <child link="{SHORT_TIP_LINK}"/>
  </joint>
"""


def tilt_rotation(tilt):
    """Mount rotation: local +Z points into the sampled cone."""
    from transforms3d.euler import euler2mat
    theta, azimuth = np.radians(tilt)
    return euler2mat(0, 0, azimuth) @ euler2mat(0, theta, 0) @ euler2mat(0, 0, -azimuth)


def _tilted_links(fraction, tilt):
    from transforms3d.euler import mat2euler
    kept = STICK_LENGTH_M * fraction
    rpy = ' '.join(f'{v:.12f}' for v in mat2euler(tilt_rotation(tilt)))
    return f'''
  <link name="{LOWER_LINK}">
{_cylinder('visual', STICK_START_M + STICK_LENGTH_M / 2, STICK_LENGTH_M)}
{_cylinder('collision', STICK_START_M + STICK_LENGTH_M / 2, STICK_LENGTH_M)}
  </link>
  <joint name="{LOWER_LINK}_joint" type="fixed">
    <parent link="panda_hand"/><child link="{LOWER_LINK}"/>
  </joint>
  <link name="{CHANGED_LINK}">
{_cylinder('visual', kept / 2, kept)}
{_cylinder('collision', kept / 2, kept)}
  </link>
  <joint name="{CHANGED_LINK}_joint" type="fixed">
    <origin xyz="0 0 {STICK_START_M:.9f}" rpy="{rpy}"/>
    <parent link="panda_hand"/><child link="{CHANGED_LINK}"/>
  </joint>
  <link name="{SHORT_TIP_LINK}"/>
  <joint name="{SHORT_TIP_LINK}_joint" type="fixed">
    <origin xyz="0 0 {kept:.9f}" rpy="0 0 0"/>
    <parent link="{CHANGED_LINK}"/><child link="{SHORT_TIP_LINK}"/>
  </joint>
'''


def robot_uid(fraction, tilt=None):
    """Keep different episode geometries isolated in the registry and cache."""
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("The kept stick fraction must be between zero and one")
    identity = float(fraction).hex()
    if tilt is not None:
        if len(tilt) != 2 or not np.isfinite(tilt).all() or not 0 < tilt[0] <= 30:
            raise ValueError('Stick tilt must have a polar angle above 0 and at most 30 degrees')
        identity += '/tilted-v2/' + '/'.join(float(v).hex() for v in tilt)
    key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return f"{UID}_{key}"


def asset_paths(fraction, tilt=None):
    """Write the split-stick URDF/SRDF to a user cache directory; return the URDF path."""
    from mani_skill.agents.robots.panda.panda_stick import PandaStick

    source = Path(PandaStick.urdf_path)
    cache = Path.home() / ".cache" / "robomme_plus"
    uid = robot_uid(fraction, tilt)
    urdf, srdf = cache / f"{uid}.urdf", cache / f"{uid}.srdf"
    text = source.read_text()
    for block in (_FULL, _FULL_COLLISION):
        if text.count(block) != 1:
            raise RuntimeError(
                "Unexpected panda_stick.urdf layout; cannot split the stick"
            )
    kept = STICK_LENGTH_M * fraction
    upper_center = STICK_START_M + kept / 2
    text = (
        text.replace(_FULL, _cylinder("visual", upper_center, kept) if tilt is None else '')
        .replace(_FULL_COLLISION, _cylinder("collision", upper_center, kept) if tilt is None else '')
        .replace(
            '  <link name="panda_hand_tcp"/>',
            (_lower_links(fraction) if tilt is None else _tilted_links(fraction, tilt)) + '  <link name="panda_hand_tcp"/>',
            1,
        )
    )
    # Mesh paths stay relative (the motion planner resolves them against the
    # URDF's directory); link each mesh folder into the cache directory.
    folders = sorted(
        {name.split("/")[0] for name in re.findall(r'filename="(?!/)([^"]+)"', text)}
    )
    semantic = source.with_suffix(".srdf").read_text()
    pairs = [
        line.replace('"panda_hand"', f'"{LOWER_LINK}"')
        for line in semantic.splitlines()
        if "<disable_collisions" in line and '"panda_hand"' in line
    ]
    pairs.append(
        f'    <disable_collisions link1="panda_hand" link2="{LOWER_LINK}" reason="Adjacent"/>'
    )
    if tilt is not None:
        pairs += [line.replace(LOWER_LINK, CHANGED_LINK) for line in pairs.copy()]
        pairs.append(f'    <disable_collisions link1="{LOWER_LINK}" link2="{CHANGED_LINK}" reason="Adjacent"/>')
    semantic = semantic.replace("</robot>", "\n".join(pairs) + "\n</robot>")
    cache.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        if not (cache / folder).exists():
            try:
                (cache / folder).symlink_to(source.parent / folder, target_is_directory=True)
            except FileExistsError:
                pass  # Another evaluation process created the same mesh link.
    for path, content in ((urdf, text), (srdf, semantic)):
        if not path.exists() or path.read_text() != content:
            # Publish complete assets even when several workers replay a seed.
            with tempfile.NamedTemporaryFile(mode="w", dir=cache, delete=False) as handle:
                handle.write(content)
                temporary = Path(handle.name)
            temporary.replace(path)
    return str(urdf)


def register(fraction, tilt=None):
    """Register the robot (idempotent). Like panda_stick, it uses the default camera configuration."""
    from mani_skill.agents.registration import REGISTERED_AGENTS, register_agent
    from mani_skill.agents.robots.panda.panda_stick import PandaStick

    uid = robot_uid(fraction, tilt)
    if uid not in REGISTERED_AGENTS:
        register_agent()(
            type(
                "PandaSplitStick",
                (PandaStick,),
                {"uid": uid, "urdf_path": asset_paths(fraction, tilt)},
            )
        )
    return uid


def links(env):
    by_name = {link.name: link for link in env.agent.robot.get_links()}
    env._stick_changed = by_name.get(CHANGED_LINK)
    return by_name[LOWER_LINK], by_name[SHORT_TIP_LINK]


def set_short(env, short):
    """Hide (short) or restore the lower part: visual and collisions, CPU simulation."""
    import sapien

    shapes = [
        shape
        for body in env._stick_lower._bodies
        for shape in body.get_collision_shapes()
    ]
    if not hasattr(env, "_stick_collision_groups"):
        env._stick_collision_groups = [
            list(shape.get_collision_groups()) for shape in shapes
        ]
    for shape, groups in zip(shapes, env._stick_collision_groups):
        shape.set_collision_groups([0, 0, 0, 0] if short else groups)
    for body in env._stick_lower._bodies:
        render = body.entity.find_component_by_type(sapien.render.RenderBodyComponent)
        if render is not None:
            render.visibility = 0.0 if short else 1.0
    changed = getattr(env, '_stick_changed', None)
    if changed is not None:
        shapes = [shape for body in changed._bodies for shape in body.get_collision_shapes()]
        if not hasattr(env, '_stick_changed_collision_groups'):
            env._stick_changed_collision_groups = [list(shape.get_collision_groups()) for shape in shapes]
        for shape, groups in zip(shapes, env._stick_changed_collision_groups):
            shape.set_collision_groups(groups if short else [0, 0, 0, 0])
        for body in changed._bodies:
            render = body.entity.find_component_by_type(sapien.render.RenderBodyComponent)
            if render is not None:
                render.visibility = 1.0 if short else 0.0
    env._stick_short = bool(short)


def scoring_point(env, obj):
    """The real stick end replaces the TCP in scoring while the stick is short."""
    if getattr(env, "_stick_short", False) and obj is env.agent.tcp:
        return env._stick_short_tip
    return obj


def tip_position(env):
    """Where the stick's real end is (the TCP while the stick is full length)."""
    from robomme_plus.common.sim import position

    return position(scoring_point(env, env.agent.tcp), False)


def tip_angle_deg(env):
    """Angle of the physical stick axis from world downward vertical."""
    from transforms3d.quaternions import quat2mat
    actor = scoring_point(env, env.agent.tcp)
    q = actor.pose.q.detach().cpu().numpy().reshape(-1, 4)[0]
    axis = quat2mat(q)[:, 2]
    return float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1))))


def drawing_allowed(env):
    drawing = getattr(env, '_drawing', None)
    if drawing is None:
        return True
    angle = tip_angle_deg(env) if drawing.max_angle_deg is not None else None
    return drawing.can_draw(tip_position(env), angle)


def patch_scoring(module):
    """Route a native tracing module's scoring and trail through the real tip and the drawing rules.

    ``is_obj_swing_onto`` scores the real tip, which must also be drawing (at most
    the drawing height above the table). ``highlight_position`` (the native white
    trail) spawns a bubble only on drawing steps, at the real tip; every bubble
    keeps its own clock.
    """
    original_swing, original_trail = module.is_obj_swing_onto, module.highlight_position
    if getattr(original_swing, "_split_stick", False):
        return

    def is_obj_swing_onto(self, obj, target, *args, **kwargs):
        drawing = getattr(self, "_drawing", None)
        if drawing is not None and obj is self.agent.tcp:
            if not drawing_allowed(self):
                return False
            kwargs["z_threshold"] = min(
                kwargs.get("z_threshold", 0.1), drawing.height_m
            )
        return original_swing(self, scoring_point(self, obj), target, *args, **kwargs)

    def highlight_position(
        self, position, start_step, end_step, cur_step, *args, **kwargs
    ):
        drawing = getattr(self, "_drawing", None)
        if drawing is None:
            return original_trail(
                self, position, start_step, end_step, cur_step, *args, **kwargs
            )
        tip = tip_position(self)
        angle = tip_angle_deg(self) if drawing.max_angle_deg is not None else None
        drawing_tip = tip.copy()
        to_rest = getattr(self, "_route_to_rest", None)
        if callable(to_rest):
            drawing_tip[:2] = to_rest(tip[:2])
        if drawing.update(int(cur_step), drawing_tip, angle):
            return original_trail(
                self, tip, start_step, end_step, cur_step, *args, **kwargs
            )
        # Not drawing: update existing bubbles only (no new one this step).
        return original_trail(
            self, position, int(cur_step) + 1, end_step, cur_step, *args, **kwargs
        )

    is_obj_swing_onto._split_stick = True
    module.is_obj_swing_onto = is_obj_swing_onto
    module.highlight_position = highlight_position


def lowered(pose, fraction, tilt=None):
    """Planner goal for the TCP that puts the real (short) tip at ``pose``."""
    import sapien
    from transforms3d.quaternions import quat2mat

    if fraction == 1 and tilt is None:
        return pose
    if tilt is not None:
        from transforms3d.quaternions import mat2quat
        mount = tilt_rotation(tilt)
        hand = quat2mat(np.asarray(pose.q)) @ mount.T
        offset = np.array([0, 0, STICK_LENGTH_M]) - mount @ [0, 0, STICK_LENGTH_M * fraction]
        return sapien.Pose(np.asarray(pose.p) + hand @ offset, mat2quat(hand))
    removed = STICK_LENGTH_M * (1 - fraction)
    p = np.asarray(pose.p) + quat2mat(np.asarray(pose.q)) @ [0, 0, removed]
    return sapien.Pose(p, pose.q)
