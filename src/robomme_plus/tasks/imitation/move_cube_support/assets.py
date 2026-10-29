"""Physical tools, straight ramp and passive mounted cue."""

from pathlib import Path
import numpy as np
import sapien
from scipy.spatial import ConvexHull
from transforms3d.euler import euler2quat
from robomme_plus.tasks.imitation.utils import tools as native_tools
from robomme_plus.tasks.imitation.utils.tools import (
    LEGAL_CONTACT_COLOR,
    SHAFT_COLOR,
    HANDLE_COLOR,
)
from .layout import SLOT, CUE_TIP_RADIUS


def build_dataset_tool(env, params, initial_pose, name):
    import sapien
    from transforms3d.euler import euler2quat

    method, length = env._imitation_config["method"], params["tool_length_m"]
    contact_color = sapien.render.RenderMaterial(
        base_color=LEGAL_CONTACT_COLOR, roughness=0.65
    )
    shaft_color = sapien.render.RenderMaterial(base_color=SHAFT_COLOR, roughness=0.65)
    handle_color = sapien.render.RenderMaterial(base_color=HANDLE_COLOR, roughness=0.65)
    builder = env.scene.create_articulation_builder()
    builder.initial_pose = initial_pose
    handle = builder.create_link_builder()
    handle.set_name("imitation_handle")
    handle_half = params["handle_half_size_m"]
    head_scale = params.get("hook_head_scale", 1.0)
    pan_half = params.get("pan_length_m", 0.11) / 2
    pan_width_half = params.get("pan_width_m", 0.10) / 2
    handle.add_box_collision(
        half_size=handle_half,
        density=650,
        material=sapien.physx.PhysxMaterial(
            static_friction=2.0, dynamic_friction=2.0, restitution=0
        ),
    )
    handle.add_box_visual(
        half_size=handle_half,
        material=handle_color,
    )

    def link(name, offset):
        part = builder.create_link_builder(handle)
        part.set_name(name)
        part.set_joint_name(name + "_fixed")
        part.set_joint_properties(
            type="fixed",
            limits=[[0, 0]],
            pose_in_parent=sapien.Pose(offset),
            pose_in_child=sapien.Pose(),
            friction=0,
            damping=0,
        )
        return part

    shaft = link("imitation_shaft", [0, 0, 0])
    radius = params.get("shaft_radius_m", 0.006)
    shaft_length = length - pan_half if method == "scoop" else length
    shaft.add_box_collision(
        sapien.Pose([shaft_length / 2, 0, 0]),
        [shaft_length / 2, radius, radius],
        density=400,
    )
    shaft.add_box_visual(
        sapien.Pose([shaft_length / 2, 0, 0]),
        [shaft_length / 2, radius, radius],
        material=shaft_color,
    )
    offset = [
        length,
        0.060 * head_scale if method == "hook" else 0,
        -0.012 if method == "scoop" else 0,
    ]
    head = link("imitation_contact_head", offset)
    if method == "poke":
        head.add_sphere_collision(radius=params["ball_radius_m"], density=500)
        head.add_sphere_visual(radius=params["ball_radius_m"], material=contact_color)
    elif method == "hook":
        head.add_box_collision(
            sapien.Pose(np.array([0, -0.016, 0]) * head_scale),
            np.array([0.006, 0.050, 0.008]) * head_scale,
            density=500,
        )
        head.add_box_visual(
            sapien.Pose(np.array([0, -0.016, 0]) * head_scale),
            np.array([0.006, 0.050, 0.008]) * head_scale,
            material=contact_color,
        )
        # Short return tips keep the cube inside the hook during a diagonal
        # pull. They belong to the coral contact head, clear of the blue-gray shaft.
        for y in (-0.037, 0.037):
            head.add_box_collision(
                sapien.Pose(np.array([-0.025, y, 0]) * head_scale),
                np.array([0.025, 0.004, 0.008]) * head_scale,
                density=500,
            )
            head.add_box_visual(
                sapien.Pose(np.array([-0.025, y, 0]) * head_scale),
                np.array([0.025, 0.004, 0.008]) * head_scale,
                material=contact_color,
            )
    else:
        # Thin floor with a sloped leading edge; three rims leave the front open
        # for loading and tipping. Every rendered surface also collides.
        assets = Path(native_tools.__file__).with_name("assets")
        if "pan_lip_thickness_m" in params:
            tip_z = -0.002 + params["pan_lip_thickness_m"]
            local_assets = env.mesh_dir / "pan_assets"
            local_assets.mkdir(exist_ok=True)
            for asset in (
                "imitation_pan_floor.obj",
                "imitation_pan_floor_inner.obj",
                "imitation_pan_floor_underside.obj",
                "imitation_pan_floor_outer.obj",
            ):
                (local_assets / asset).write_text(
                    (assets / asset).read_text().replace("-0.0018", f"{tip_z:.8f}")
                )
            floor_assets = local_assets
        else:
            floor_assets = assets
        floor = str(floor_assets / "imitation_pan_floor.obj")
        head.add_convex_collision_from_file(
            filename=floor,
            scale=[pan_half / 0.055, pan_width_half / 0.050, 1],
            density=450,
        )
        pan_surfaces = (
            ("inner", contact_color),
            ("underside", contact_color),
            ("outer", shaft_color),
        )
        for surface, material in pan_surfaces:
            head.add_visual_from_file(
                filename=str(floor_assets / f"imitation_pan_floor_{surface}.obj"),
                scale=[pan_half / 0.055, pan_width_half / 0.050, 1],
                material=material,
            )
        rim_height = params.get("pan_rim_height_m", 0.030)
        rim_z = rim_height / 2 - 0.002
        side_front = params.get("pan_rim_front_m", pan_half)
        side_center = (side_front - pan_half) / 2
        side_half = (side_front + pan_half) / 2
        parts = [
            ([-pan_half + 0.002, 0, rim_z], [0.002, pan_width_half, rim_height / 2], 0),
            (
                [side_center, -pan_width_half, rim_z],
                [side_half, 0.002, rim_height / 2],
                0,
            ),
            (
                [side_center, pan_width_half, rim_z],
                [side_half, 0.002, rim_height / 2],
                0,
            ),
        ]
        for index, (p, half, pitch) in enumerate(parts):
            pose = sapien.Pose(p, euler2quat(0, pitch, 0))
            head.add_box_collision(pose, half, density=450)
            # Split the visual faces at the real collision surface. The back
            # rim's inner normal is +x; the side rims point +y and -y inward.
            yaw = 0 if index == 0 else np.pi / 2 if index == 1 else -np.pi / 2
            scale = half if index == 0 else [half[1], half[0], half[2]]
            for surface, material in pan_surfaces:
                head.add_visual_from_file(
                    filename=str(assets / f"imitation_pan_rim_{surface}.obj"),
                    pose=sapien.Pose(p, euler2quat(0, 0, yaw)),
                    scale=scale,
                    material=material,
                )
    tool = builder.build(name=name, fix_root_link=False)
    links = {item.name: item for item in tool.get_links()}
    if method == "scoop" and "pan_floor_rest_offset_m" in params:
        floor = links["imitation_contact_head"]._bodies[0].get_collision_shapes()[0]
        floor.set_contact_offset(0.001)
        floor.set_rest_offset(params["pan_floor_rest_offset_m"])
    return tool, links


def write_convex(path, vertices):
    v = np.asarray(vertices)
    hull = ConvexHull(v)
    lines = ["v " + " ".join(f"{x:.9f}" for x in p) for p in v]
    for face, equation in zip(hull.simplices, hull.equations):
        a, b, c = face
        if np.dot(np.cross(v[b] - v[a], v[c] - v[a]), equation[:3]) < 0:
            b, c = c, b
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    path.write_text("\n".join(lines) + "\n")


def build_ramp(self):
    l = self.layout
    b = self.scene.create_actor_builder()
    b.initial_pose = sapien.Pose()
    slot = SLOT if l.ramp_slot_m is None else l.ramp_slot_m
    self.ramp_collision_vertices = []
    mat = sapien.physx.PhysxMaterial(0.8, 0.8, 0)

    def part(name, vertices, color):
        path = self.mesh_dir / (name + ".obj")
        write_convex(path, vertices)
        b.add_convex_collision_from_file(str(path), material=mat)
        b.add_visual_from_file(str(path), material=color)
        self.ramp_collision_vertices.append(np.asarray(vertices))

    for i in range(20):
        ts = (i / 20, (i + 1) / 20)
        vertices = [
            l.point(t, w, z)
            for t in ts
            for w in (-l.ramp_width_m / 2, l.ramp_width_m / 2)
            for z in (min(slot, l.ramp_height * t - 0.008), l.ramp_height * t)
        ]
        part(f"body_{i:02}", vertices, [0.20, 0.49, 0.69, 1])
        # A central support belongs to the same ramp body. Both side edges
        # and the raised end retain a floor-level pan-loading undercut.
        if 5 <= i <= 14:
            feet = [
                l.point(t, w, z)
                for t in ts
                for w in (-0.018, 0.018)
                for z in (-0.004, max(0.002, l.ramp_height * t - 0.006))
            ]
            part(f"foot_{i:02}", feet, [0.17, 0.40, 0.55, 1])
        if i % 2 == 0 and i:
            pose = sapien.Pose(
                l.point(i / 20) + [0, 0, 0.0006],
                euler2quat(0, l.ramp_angle, np.radians(l.ramp_heading_deg)),
            )
            b.add_box_visual(
                pose=pose,
                half_size=[0.0006, l.ramp_width_m / 2 - 0.003, 0.0004],
                material=[0.78, 0.9, 0.95, 1],
            )
    self.ramp = b.build_kinematic(self._build_phase + "_ramp")


def build_cue(self):
    # Passive variable-travel cue. The robot supplies every motion.
    l = self.layout
    b = self.scene.create_articulation_builder()
    b.initial_pose = sapien.Pose(l.cue_pivot)
    b.disable_self_collisions = True
    root = b.create_link_builder()
    root.set_name("cue_fixed_mount")
    root.add_box_collision(
        pose=sapien.Pose([0, 0, -0.01]), half_size=[0.025, 0.025, 0.01]
    )
    root.add_box_visual(
        pose=sapien.Pose([0, 0, -0.01]),
        half_size=[0.025, 0.025, 0.01],
        material=[0.16, 0.23, 0.29, 1],
    )
    yaw = b.create_link_builder(root)
    yaw.set_name("cue_yaw_bearing")
    yaw.set_joint_name("cue_aim")
    frame = sapien.Pose(q=euler2quat(0, -np.pi / 2, 0))
    # Allow an aiming arc to cross +/- pi without encountering an artificial
    # stop. This passive bearing is rotational; only axial travel is bounded.
    yaw.set_joint_properties(
        type="revolute",
        limits=[[-4 * np.pi, 4 * np.pi]],
        pose_in_parent=frame,
        pose_in_child=frame,
        friction=0.015,
        damping=l.cue_yaw_damping,
    )
    for s in (-1, 1):
        pose = sapien.Pose([0, s * 0.017, 0.008])
        yaw.add_box_collision(pose=pose, half_size=[0.018, 0.005, 0.018])
        yaw.add_box_visual(
            pose=pose, half_size=[0.018, 0.005, 0.018], material=[0.3, 0.53, 0.66, 1]
        )
    cue = b.create_link_builder(yaw)
    cue.set_name("ring_sliding_cue")
    cue.set_joint_name("ring_stroke")
    cue.set_joint_properties(
        type="prismatic",
        limits=[[0, l.cue_stroke_m]],
        pose_in_parent=sapien.Pose(),
        pose_in_child=sapien.Pose(),
        friction=0,
        damping=0,
    )
    rod = sapien.Pose([(l.cue_front_m - l.cue_rear_m) / 2, 0, 0])
    half = [(l.cue_front_m + l.cue_rear_m) / 2, 0.006, 0.006]
    cue.add_box_collision(pose=rod, half_size=half, density=1200)
    cue.add_box_visual(pose=rod, half_size=half, material=[0.72, 0.48, 0.24, 1])
    cue.add_sphere_collision(
        pose=sapien.Pose([l.cue_front_m, 0, 0]),
        radius=CUE_TIP_RADIUS,
        density=1000,
        material=sapien.physx.PhysxMaterial(0.01, 0.01, 0),
    )
    cue.add_sphere_visual(
        pose=sapien.Pose([l.cue_front_m, 0, 0]),
        radius=CUE_TIP_RADIUS,
        material=[0.13, 0.57, 0.74, 1],
    )
    handle = sapien.Pose([-l.cue_grip_offset_m, 0, 0.06])
    half = [0.012, 0.012, 0.06]
    cue.add_box_collision(
        pose=handle,
        half_size=half,
        density=650,
        material=sapien.physx.PhysxMaterial(1.8, 1.8, 0),
    )
    cue.add_box_visual(pose=handle, half_size=half, material=[0.19, 0.26, 0.31, 1])
    self.cue = b.build(name=self._build_phase + "_mounted_cue", fix_root_link=True)
    self.cue_link = next(
        x for x in self.cue.get_links() if x.name == "ring_sliding_cue"
    )
    for index, joint in enumerate(self.cue.get_active_joints()):
        joint.set_drive_properties(
            stiffness=0, damping=l.cue_yaw_damping if index == 0 else 0
        )
