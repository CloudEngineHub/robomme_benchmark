"""Native scene construction and per-physics-step contact collection."""

import copy
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from mani_skill.utils.structs.types import SimConfig, SceneConfig
from robomme_plus.simulation.base import MemoryTask
from robomme_plus.simulation.actors import build_purple_white_target
from robomme_plus.control.reset import get_reset_panda_param
from robomme_plus.evaluation.timing import CONTROL_FREQUENCY
from robomme_plus.tasks.imitation.utils.tools import empty_evidence
from robomme_plus.tasks.imitation.utils.controllers import rotation
from .layout import HALF, TARGET_RADIUS, METHODS, TOL, unit
from .assets import build_dataset_tool as build_tool_articulation, build_ramp, build_cue
from .pan import classify_pan_contact, record_pan_contacts, PAN_SCORING_VERSION

REST_QPOS = get_reset_panda_param("qpos")


def pos(actor):
    return actor.pose.p[0].detach().cpu().numpy().copy()


class Scene(MemoryTask):
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    cube_half_size = HALF
    build_ramp = build_ramp
    build_cue = build_cue

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=500,
            control_freq=CONTROL_FREQUENCY,
            scene_config=SceneConfig(
                contact_offset=0.0005,
                solver_position_iterations=30,
                solver_velocity_iterations=8,
                enable_ccd=True,
            ),
        )

    def _load_scene(self, options):
        l = self.layout
        b = self.scene.create_actor_builder()
        b.initial_pose = sapien.Pose(
            l.cube, euler2quat(0, 0, np.radians(l.target_heading_deg))
        )
        b.add_box_collision(
            half_size=[HALF] * 3,
            density=l.cube_mass_kg / 0.04**3,
            material=sapien.physx.PhysxMaterial(0.06, 0.06, 0),
        )
        b.add_box_visual(half_size=[HALF] * 3, material=[1, 0, 0, 1])
        self.cube = b.build(self._build_phase + "_cube")
        self.goal_site = build_purple_white_target(
            self.scene,
            radius=TARGET_RADIUS,
            thickness=0.005,
            name=self._build_phase + "_target",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(
                [*l.target[:2], 0.005], euler2quat(0, np.pi / 2, 0)
            ),
        )
        self.handheld = {}
        for method, params in l.tool_params.items():
            if method not in self.available_methods:
                continue
            self._imitation_config = {"method": method, "layout_version": 6}
            p, yaw = l.tool_poses[method]
            pose = sapien.Pose(p, euler2quat(0, 0, np.radians(yaw)))
            tool, links = build_tool_articulation(
                self, params, pose, self._build_phase + "_" + method
            )
            if method == "scoop" and l.pan_floor_friction is not None:
                floor = (
                    links["imitation_contact_head"]._bodies[0].get_collision_shapes()[0]
                )
                floor.set_physical_material(
                    sapien.physx.PhysxMaterial(
                        l.pan_floor_friction, l.pan_floor_friction, 0
                    )
                )
            self.handheld[method] = dict(
                tool=tool,
                head=links["imitation_contact_head"],
                handle=links["imitation_handle"],
                shaft=links["imitation_shaft"],
                params=params,
            )
        self.ramp = self.cue = self.cue_link = self.ball = None
        if set(self.available_methods) & {"ball", "scoop"}:
            self.build_ramp()
        if "cue" in self.available_methods:
            self.build_cue()
        if "ball" in self.available_methods:
            b = self.scene.create_actor_builder()
            b.initial_pose = sapien.Pose(l.ball_rest_xyz)
            b.add_sphere_collision(
                radius=l.ball_radius,
                density=l.ball_mass / (4 * np.pi * l.ball_radius**3 / 3),
                material=sapien.physx.PhysxMaterial(0.12, 0.12, 0),
            )
            b.add_sphere_visual(radius=l.ball_radius, material=[1, 0.68, 0.08, 1])
            for axis in np.eye(3):
                for sign in (-1, 1):
                    b.add_sphere_visual(
                        pose=sapien.Pose(axis * sign * (l.ball_radius - 0.001)),
                        radius=0.004,
                        material=[0.14, 0.16, 0.18, 1],
                    )
            self.ball = b.build(self._build_phase + "_ball")
            self.ball._bodies[0].set_linear_damping(0.15)
            self.ball._bodies[0].set_angular_damping(0.8)
        self.robot_entities = {
            a._bodies[0].entity for a in self.agent.robot.get_links()
        }
        self.contact_labels = {}
        for label, actor in (("ball", self.ball), ("ramp", self.ramp)):
            if actor is not None:
                self.contact_labels[actor._bodies[0].entity] = label
        if self.cue is not None:
            for link in self.cue.get_links():
                self.contact_labels[link._bodies[0].entity] = (
                    "cue" if link.name == self.cue_link.name else "cue_mount"
                )
        for method, e in self.handheld.items():
            for part in ("head", "shaft", "handle"):
                self.contact_labels[e[part]._bodies[0].entity] = method + "_" + part

    def _initialize_episode(self, env_idx, options):
        l = self.layout
        self.table_scene.initialize(env_idx)
        self.agent.reset(np.array(REST_QPOS))
        yaw = l.target_heading_deg if l.cube_yaw_deg is None else l.cube_yaw_deg
        self.cube.set_pose(sapien.Pose(l.cube, euler2quat(0, 0, np.radians(yaw))))
        self.cube.set_linear_velocity([0, 0, 0])
        self.cube.set_angular_velocity([0, 0, 0])
        if self.ball is not None:
            self.ball.set_pose(sapien.Pose(l.ball_rest_xyz))
            self.ball.set_linear_velocity([0, 0, 0])
            self.ball.set_angular_velocity([0, 0, 0])
        for method, e in self.handheld.items():
            p, yaw = l.tool_poses[method]
            e["tool"].set_pose(sapien.Pose(p, euler2quat(0, 0, np.radians(yaw))))
            e["tool"].set_root_linear_velocity([0, 0, 0])
            e["tool"].set_root_angular_velocity([0, 0, 0])
        if self.cue is not None:
            parked = (l.cue_park_deg + 180) % 360 - 180
            self.cue.set_pose(sapien.Pose(l.cue_pivot))
            self.cue.set_qpos(np.array([np.radians(parked), 0.0]))
            self.cue.set_qvel(np.zeros(2))
        self.clear_evidence()

    def _get_obs_extra(self, info):
        # Fixed shape with explicit presence mask; no requested-method label.
        zero = torch.zeros_like(self.cube.pose.raw_pose)
        return dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            cube_pose=self.cube.pose.raw_pose,
            target_pose=self.goal_site.pose.raw_pose,
            available_methods=torch.tensor(
                [[m in self.available_methods for m in METHODS]], device=self.device
            ),
            ball_pose=self.ball.pose.raw_pose if self.ball is not None else zero,
            cue_qpos=self.cue.get_qpos() if self.cue is not None else zero[:, :2],
            **{
                m + "_pose": self.handheld[m]["tool"].pose.raw_pose
                if m in self.handheld
                else zero
                for m in METHODS[:3]
            },
        )

    def _after_simulation_step(self):
        if not hasattr(self, "events"):
            return
        self.time += 1 / self.sim_freq
        self._trace_step += 1
        cube_entity = self.cube._bodies[0].entity
        ball_entity = self.ball._bodies[0].entity if self.ball is not None else None
        cue_entity = self.cue_link._bodies[0].entity if self.cue is not None else None
        pan_entity = (
            self.handheld["scoop"]["head"]._bodies[0].entity
            if "scoop" in self.handheld
            else None
        )
        current = set()
        ball_robot = False
        cue_robot = False
        pan_regions = set()
        for c in self.scene.px.get_contacts():
            points = [p for p in c.points if np.linalg.norm(p.impulse) > 1e-7]
            if not points:
                continue
            pair = {b.entity for b in c.bodies}
            robot = bool(pair & self.robot_entities)
            ball_robot |= ball_entity in pair and robot
            cue_robot |= cue_entity in pair and robot
            if self.ramp is not None and pair == {
                ball_entity,
                self.ramp._bodies[0].entity,
            }:
                self.events["ball_ramp"] = True
            if cube_entity not in pair:
                continue
            if robot:
                self.events["robot_cube"] = True
            current.update(
                self.contact_labels[e] for e in pair if e in self.contact_labels
            )
            if pan_entity in pair:
                head = self.handheld["scoop"]["head"]
                r = rotation(head)
                origin = pos(head)
                sign = 1 if c.bodies[0].entity == cube_entity else -1
                for p in points:
                    pan_regions.add(
                        classify_pan_contact(
                            r.T @ (np.asarray(p.position) - origin),
                            r.T @ (sign * np.asarray(p.normal)),
                            self.handheld["scoop"]["params"],
                        )
                    )
        location = pos(self.cube)
        delta = location - self.last_cube
        self.last_cube = location
        speed = float(np.linalg.norm(self.cube._bodies[0].linear_velocity[:2]))
        self.peak_cube_speed = max(self.peak_cube_speed, speed)
        self.events["cube_lifted"] |= bool(location[2] > HALF + 0.025)
        self.events["ball_grasped"] |= ball_robot and bool(
            self.agent.is_grasping(self.ball).item()
        )
        self.events["cue_grasped"] |= cue_robot and bool(
            self.agent.is_grasping(self.cue_link).item()
        )
        if "ball" in current:
            self.events["ball_free_hit"] |= not ball_robot
            self.events["ball_held_hit"] |= ball_robot
        self.events["cue_hit"] |= "cue" in current
        q = (
            self.cue.get_qpos()[0].cpu().numpy()
            if self.cue is not None
            else np.zeros(2)
        )
        self.cue_limits = [
            min(self.cue_limits[0], float(q[1])),
            max(self.cue_limits[1], float(q[1])),
        ]
        if self.cue is not None:
            self.mount_drift = max(
                self.mount_drift,
                float(np.linalg.norm(pos(self.cue) - self.layout.cue_pivot)),
            )
        if "cue" in current:
            self.events["contact_outside_cue"] |= bool(
                np.linalg.norm((location - self.layout.cue_pivot)[:2])
                > self.layout.cue_contact_radius + 0.001
            )
        for label in current:
            self.cube_contacts.add(label)
            if label not in self.first_contact:
                self.first_contact[label] = self.time
                self.impact_speed[label] = (
                    float(np.linalg.norm(self._previous_ball_velocity))
                    if label == "ball"
                    else float(self.cue.get_qvel()[0, 1])
                    if label == "cue"
                    else speed
                )
            self.last_contact[label] = dict(t=self.time, xyz=location.tolist())
        direction = unit(self.layout.target_heading_deg)
        for method, entry in self.handheld.items():
            d = self.evidence[method]
            hit = method + "_head" in current
            d["robot_contact_steps"] += int(self.events["robot_cube"])
            d["shaft_contact_steps"] += int(method + "_shaft" in current)
            d["handle_contact_steps"] += int(method + "_handle" in current)
            d["cube_lifted"] |= self.events["cube_lifted"]
            if (
                method == "scoop"
                and hit
                and "ramp" in current
                and bool(self.agent.is_grasping(entry["handle"]).item())
            ):
                d["wall_contact_steps"] += 1
            if not hit:
                continue
            held = bool(self.agent.is_grasping(entry["handle"]).item())
            local = rotation(entry["head"]).T @ (location - pos(entry["head"]))
            if method == "scoop":
                record_pan_contacts(
                    d,
                    pan_regions,
                    step=self._trace_step,
                    cube=location,
                    head=pos(entry["head"]),
                )
                supported = (
                    held
                    and "floor" in pan_regions
                    and d["wall_contact_steps"] >= 2
                    and abs(local[0]) < entry["params"]["pan_length_m"] / 2
                    and abs(local[1]) < entry["params"]["pan_width_m"] / 2
                    and local[2] > 0
                    and location[2] > HALF + 0.04
                )
                d["supported_lift_steps"] += int(supported)
                if supported and np.linalg.norm(delta) < 0.04:
                    d["carried_distance_m"] += float(np.linalg.norm(delta[:2]))
            else:
                correct = (
                    method == "poke" and "poke_shaft" not in current
                    and "poke_handle" not in current
                ) or (
                    method == "hook" and held and (
                        rotation(entry["head"])[:, 0] @ direction < -0.8
                        and local[0] < -0.006
                        and abs(local[1]) < 0.029
                    )
                )
                if correct and np.linalg.norm(delta) < 0.04:
                    d["correct_contact_steps"] += 1
                    d["forward_motion_m"] += max(0.0, float(delta @ direction))
        near = np.linalg.norm((location - self.layout.target)[:2]) <= TOL
        still = (
            speed < 0.015
            and np.linalg.norm(self.cube._bodies[0].angular_velocity) < 0.1
            and abs(location[2] - HALF) < 0.008
        )
        self.stable_steps = self.stable_steps + 1 if near and still else 0
        self.current_contacts = current
        if self.ball is not None:
            # The current physics step has already applied collision impulses.
            # Save its velocity for the next step's incoming-impact evidence.
            self._previous_ball_velocity = np.array(self.ball._bodies[0].linear_velocity)
        if self._trace_step % 10 == 0:
            self.trace.append(
                dict(
                    t=self.time,
                    cube=location.tolist(),
                    speed=speed,
                    contacts=sorted(current),
                    cue_slide_m=float(q[1]),
                    cue_yaw_rad=float(q[0]),
                )
            )

    def clear_evidence(self):
        self.time = 0.0
        self.stable_steps = 0
        self.trace = []
        self._trace_step = 0
        self.last_cube = pos(self.cube)
        self.evidence = {m: empty_evidence() for m in METHODS[:3]}
        self.events = dict(
            robot_cube=False,
            cube_lifted=False,
            ball_grasped=False,
            ball_ramp=False,
            ball_free_hit=False,
            ball_held_hit=False,
            cue_grasped=False,
            cue_hit=False,
            contact_outside_cue=False,
        )
        self.cube_contacts = set()
        self.last_contact = {}
        self.first_contact = {}
        self.impact_speed = {}
        self._previous_ball_velocity = (
            np.array(self.ball._bodies[0].linear_velocity)
            if self.ball is not None else np.zeros(3)
        )
        self.cue_limits = [float("inf"), -float("inf")]
        self.mount_drift = 0.0
        self.peak_cube_speed = 0.0
        self._imitation_evidence = {
            "execution": empty_evidence(),
            "demonstration": empty_evidence(),
        }

        self._limit_diagnostics = {
            "reach": dict(
                shoulder_xyz_m=[-0.615, 0, 0.333],
                finger_extent_allowance_m=0,
                conservative_tcp_radius_m=0.83,
            ),
            "trace": [],
        }
        self.current_contacts = set()

    def activate_handheld(self, method):
        self.active_oracle = method
        e = self.handheld[method]
        self._imitation_head = e["head"]
        self._imitation_handle = e["handle"]
        self._imitation_shaft = e["shaft"]
        self._imitation_tool = e["params"]
        self._imitation_config = {"method": method, "layout_version": 6}
        self._imitation_evidence["execution"] = self.evidence[method]

    def result(self):
        return dict(
            success=self.method_success(self._required_method),
            pan_scoring_version=PAN_SCORING_VERSION,
            method_checks={m: self.method_success(m) for m in METHODS},
            final_error_m=float(
                np.linalg.norm((pos(self.cube) - self.layout.target)[:2])
            ),
            final_cube=pos(self.cube).tolist(),
            evidence=copy.deepcopy(self.evidence),
            events=self.events.copy(),
            cube_contacts=sorted(self.cube_contacts),
            first_contact=self.first_contact.copy(),
            last_contact=copy.deepcopy(self.last_contact),
            impact_speed=self.impact_speed.copy(),
            cue_limits=[v if np.isfinite(v) else None for v in self.cue_limits],
            mount_drift_m=self.mount_drift,
            peak_cube_speed_mps=self.peak_cube_speed,
            physics_trace=copy.deepcopy(self.trace),
        )
