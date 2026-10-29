"""Latched evidence for method-specific success and shortcut rejection."""

import copy
import numpy as np
from .scene import Scene, pos, rotation
from .layout import unit, HALF, METHODS, TARGET_RADIUS, geometry_check, ramp_gap
from .criteria import (
    verdict, VERSION, cue_impact_components, MAX_NON_HEAD_CONTACT_SECONDS,
)
from .pan import pan_evidence_is_legal
from .pan_unload import PanUnloadTracker, underside_normal, pan_touches_target
from .ramp_contact import ramp_contact_surfaces
from .behavior import (
    ToolLiftTracker, hook_inner_contact, cue_point_velocity,
    HOOK_APPROACH_RADIUS_M, MAX_DISTRACTOR_DISPLACEMENT_M,
)


class StrictScene(Scene):
    def clear_evidence(self):
        super().clear_evidence()
        self.strict = dict(
            robot_cube_contact=False,
            cube_grasped=False,
            target_moved=False,
            fixture_moved=False,
            discontinuity=False,
            pan_touched_target=False,
            tools_used=[],
            tool_max_lift_m={},
            tool_max_displacement_m={},
            wrong_tool_progress_m=0.0,
            cube_contacts=[],
            max_cube_rise_m=0.0,
            final_error_m=1.0,
            on_table=False,
            stable_target_s=0.0,
            robot_static=False,
            cube_held_now=False,
            current_contacts=[],
            ramp_surface_contact_steps={face: 0 for face in ("incline", "side_wall", "rear_wall", "edge")},
            required_flat_runout_m=0.08
            if self.ball is not None and ramp_gap(self.layout) >= 0.15
            else 0.0,
            handheld={
                m: dict(
                    unheld_cube_contact=False,
                    illegal_surface_contact=False,
                    shaft_contact_s=0.0,
                    shaft_contact_run_steps=0,
                    shaft_longest_contact_s=0.0,
                    non_head_contact_steps=0,
                    non_head_contact_run_steps=0,
                    non_head_longest_contact_s=0.0,
                    wall_contact_steps=0,
                    max_supported_rise_m=0.0,
                    longest_supported_s=0.0,
                    max_supported_carry_m=0.0,
                    max_supported_target_progress_m=0.0,
                    incorrect_contact_motion_m=0.0,
                    near_target_inner_progress_m=0.0,
                    near_target_wrong_face_motion_m=0.0,
                    underside_contact_steps=0,
                    underside_contact_motion_m=0.0,
                    correct_contact_steps=0,
                    correct_progress_m=0.0,
                )
                for m in METHODS[:3]
            },
            ball=dict(
                grasped=False,
                released_on_ramp=False,
                launched=False,
                held_hit=False,
                ramp_after_release_s=0.0,
                robot_recontact_after_release=False,
                descent_after_release_m=0.0,
                free_before_hit_s=0.0,
                rolling_rotation_rad=0.0,
                free_hit=False,
                impact_speed_mps=0.0,
                flat_runout_m=0.0,
            ),
            cue=dict(
                dynamic_hit=False,
                impact_contact_speed_mps=0.0,
                non_tip_contact=False,
                swing_hit=False,
                held_tip_hit=False,
                impact_forward_speed_mps=0.0,
                limits_respected=True,
                static_target_unreachable=False,
                coast_after_last_contact_m=0.0,
            ),
        )
        if self.cue is not None:
            try:
                self.strict["cue"]["static_target_unreachable"] = geometry_check(
                    self.layout
                )["passed"]
            except ValueError:
                pass
        self.strict_events = []
        self._strict_event_keys = set()
        self._strict_previous = pos(self.cube)
        self._strict_tool_origins = {
            m: pos(e["handle"]) for m, e in self.handheld.items()
        }
        if self.ball is not None:
            self._strict_tool_origins["ball"] = pos(self.ball)
        self._tool_lift = ToolLiftTracker()
        self._strict_head_origins = {m: pos(e["head"]) for m, e in self.handheld.items()}
        self._strict_ball_release = None
        self._strict_ball_hit = False
        self._strict_ball_last_robot_contact = -float("inf")
        self._strict_ball_separated = False
        self._strict_ball_separation_candidate = None
        self._strict_ball_ground_start = None
        self._strict_pan_start = None
        self._strict_pan_start_time = None
        self._strict_pan_last = -1.0
        self._pan_unload = PanUnloadTracker()
        self.strict["pan_unload"] = self._pan_unload.evidence
        self._strict_previous_cue_velocity = np.zeros(2)
        self._strict_cue_last_hit = -float("inf")
        self._strict_cue_strikes = 0

    def strict_event(self, kind, **details):
        key = (kind, str(details))
        if key not in self._strict_event_keys:
            self._strict_event_keys.add(key)
            self.strict_events.append(dict(t=self.time, kind=kind, **details))

    def _after_simulation_step(self):
        super()._after_simulation_step()
        if not hasattr(self, "strict"):
            return
        s = self.strict
        l = self.layout
        dt = 1 / self.sim_freq
        cube = pos(self.cube)
        delta = cube - self._strict_previous
        self._strict_previous = cube.copy()
        d = unit(l.target_heading_deg)
        current = self.current_contacts
        robot_touches = set()
        cube_entity = self.cube._bodies[0].entity
        ramp_entity = self.ramp._bodies[0].entity if self.ramp is not None else None
        ramp_surfaces = set()
        table_contact = False
        for c in self.scene.px.get_contacts():
            pair = {b.entity for b in c.bodies}
            if ramp_entity is not None and pair == {cube_entity, ramp_entity}:
                for point in c.points:
                    if np.linalg.norm(point.impulse) > 1e-7:
                        ramp_surfaces.update(ramp_contact_surfaces(l, point.position, point.normal))
            if cube_entity in pair and any(np.linalg.norm(p.impulse) > 1e-7 for p in c.points):
                other = pair - {cube_entity}
                if not other & (self.robot_entities | set(self.contact_labels)):
                    table_contact = True
            if pair & self.robot_entities and any(
                np.linalg.norm(p.impulse) > 1e-7 for p in c.points
            ):
                robot_touches.update(pair - self.robot_entities)
        for face in ramp_surfaces:
            s["ramp_surface_contact_steps"][face] += 1

        def grasped(actor):
            return (
                actor is not None
                and actor._bodies[0].entity in robot_touches
                and bool(self.agent.is_grasping(actor).item())
            )

        held = {m: grasped(e["handle"]) for m, e in self.handheld.items()}
        ball_held = grasped(self.ball)
        cue_held = grasped(self.cue_link)
        s["robot_cube_contact"] |= self.events["robot_cube"]
        s["cube_held_now"] = grasped(self.cube)
        s["cube_grasped"] |= s["cube_held_now"]
        s["target_moved"] |= bool(
            np.linalg.norm(pos(self.goal_site)[:2] - l.target[:2]) > 0.001
        )
        s["fixture_moved"] |= self.mount_drift > 0.001
        if self.ramp is not None:
            s["fixture_moved"] |= bool(np.linalg.norm(pos(self.ramp)) > 0.001)
        s["discontinuity"] |= bool(np.linalg.norm(delta) > 0.04)
        s["max_cube_rise_m"] = max(s["max_cube_rise_m"], float(cube[2] - HALF))
        s["cube_contacts"] = sorted(self.cube_contacts)
        s["current_contacts"] = sorted(current)
        s["final_error_m"] = float(np.linalg.norm((cube - l.target)[:2]))
        s["on_table"] = abs(float(cube[2]) - HALF) < 0.008
        s["stable_target_s"] = self.stable_steps / self.sim_freq
        s["robot_static"] = bool(
            np.max(np.abs(self.agent.robot.get_qvel()[0, :7].cpu().numpy())) < 0.2
        )
        def record_displacement(method, displacements):
            distance = max(float(np.linalg.norm(p[:2])) for p in displacements)
            s["tool_max_displacement_m"][method] = max(
                s["tool_max_displacement_m"].get(method, 0.0), distance,
            )
            if method != self._required_method and distance > MAX_DISTRACTOR_DISPLACEMENT_M:
                self.strict_event("distractor_moved", method=method)

        for m, e in self.handheld.items():
            handle_delta = pos(e["handle"]) - self._strict_tool_origins[m]
            head_delta = pos(e["head"]) - self._strict_head_origins[m]
            self._tool_lift.update(m, [
                handle_delta[2], head_delta[2],
            ], dt)
            record_displacement(m, [handle_delta, head_delta])
        if self.ball is not None:
            ball_delta = pos(self.ball) - self._strict_tool_origins["ball"]
            self._tool_lift.update("ball", [ball_delta[2]], dt)
            record_displacement("ball", [ball_delta])
        used = self._tool_lift.used
        s["tool_max_lift_m"] = dict(self._tool_lift.max_rise)
        required = self._required_method
        if required in ("hook", "poke", "scoop"):
            allowed = {required + suffix for suffix in ("_head", "_shaft", "_handle")}
        else:
            allowed = {required}
        if required == "scoop":
            allowed.add("ramp")
        if required == "cue":
            allowed.add("cue_mount")
        if set(current) - allowed:
            s["wrong_tool_progress_m"] += float(np.linalg.norm(delta[:2]))
        for m in sorted(used - set(s["tools_used"])):
            self.strict_event("tool_used", method=m)
        s["tools_used"] = sorted(used)
        if s["robot_cube_contact"]:
            self.strict_event("robot_cube_contact")
        if s["cube_grasped"]:
            self.strict_event("cube_grasped")
        for m, e in self.handheld.items():
            out = s["handheld"][m]
            native = self.evidence[m]
            hit = m + "_head" in current
            non_head = bool({m + "_shaft", m + "_handle"} & set(current))
            # One timer covers all non-head parts. Sliding from shaft to handle
            # without separation cannot restart the accidental-contact allowance.
            out["non_head_contact_steps"] += int(non_head)
            out["non_head_contact_run_steps"] = (
                out["non_head_contact_run_steps"] + 1 if non_head else 0
            )
            out["non_head_longest_contact_s"] = max(
                out["non_head_longest_contact_s"],
                out["non_head_contact_run_steps"] / self.sim_freq,
            )
            if out["non_head_longest_contact_s"] > MAX_NON_HEAD_CONTACT_SECONDS:
                self.strict_event("non_head_contact_too_long", method=m)
            # Keep the old shaft-only and surface flags as diagnostics. Scoring
            # uses the combined longest contact, not a cumulative contact count.
            out["shaft_contact_s"] = native["shaft_contact_steps"] / self.sim_freq
            out["shaft_contact_run_steps"] = (
                out["shaft_contact_run_steps"] + 1 if m + "_shaft" in current else 0
            )
            out["shaft_longest_contact_s"] = max(
                out["shaft_longest_contact_s"],
                out["shaft_contact_run_steps"] / self.sim_freq,
            )
            out["illegal_surface_contact"] |= bool(
                native["handle_contact_steps"]
                or (m != "poke" and native["shaft_contact_steps"])
            )
            if m == "scoop":
                out["illegal_surface_contact"] |= not pan_evidence_is_legal(native)
            if hit and not held[m]:
                out["unheld_cube_contact"] = True
                self.strict_event("unheld_head_contact", method=m)
            if out["illegal_surface_contact"]:
                self.strict_event("incidental_tool_part_contact", method=m)
            inner = self._hook_inner_only() if m == "hook" and hit else True
            hook_near_target = (
                m == "hook"
                and np.linalg.norm((cube - l.target)[:2]) <= HOOK_APPROACH_RADIUS_M
            )
            if hook_near_target and hit and not inner:
                # A tolerated shaft/handle brush must not mask simultaneous
                # pushing with an actual outer face of the hook head.
                out["near_target_wrong_face_motion_m"] += float(np.linalg.norm(delta[:2]))
            if non_head:
                if m == "scoop":
                    # A later supported carry cannot include this brush's motion
                    # or duration, even if floor contact resumes on the next step.
                    self._strict_pan_last = -1.0
                continue
            if not hit:
                continue
            r = rotation(e["head"])
            local = r.T @ (cube - pos(e["head"]))
            if m == "scoop":
                out["wall_contact_steps"] += int(held[m] and "ramp" in current)
                # The parent records floor versus inner/outer rim from actual
                # contact normals. Supported lift requires floor contact now.
                regions = self._strict_pan_regions()
                supported = (
                    held[m]
                    and "floor" in regions
                    and "ramp" not in current
                    and out["wall_contact_steps"] >= 2
                    and abs(local[0]) < e["params"]["pan_length_m"] / 2 - 0.005
                    and abs(local[1]) < e["params"]["pan_width_m"] / 2
                    and local[2] > 0
                    and cube[2] >= HALF + 0.04
                )
                if supported:
                    if self.time - self._strict_pan_last > 0.03:
                        self._strict_pan_start = cube.copy()
                        self._strict_pan_start_time = self.time
                    self._strict_pan_last = self.time
                    out["max_supported_rise_m"] = max(
                        out["max_supported_rise_m"], float(cube[2] - HALF)
                    )
                    out["longest_supported_s"] = max(
                        out["longest_supported_s"],
                        self.time - self._strict_pan_start_time + dt,
                    )
                    out["max_supported_carry_m"] = max(
                        out["max_supported_carry_m"],
                        float(np.linalg.norm((cube - self._strict_pan_start)[:2])),
                    )
                    progress = float(
                        np.linalg.norm((self._strict_pan_start - l.target)[:2])
                        - np.linalg.norm((cube - l.target)[:2])
                    )
                    out["max_supported_target_progress_m"] = max(
                        out["max_supported_target_progress_m"], progress
                    )
            else:
                underside = m == "hook" and self._hook_underside_only()
                if underside:
                    out["underside_contact_steps"] += 1
                    out["underside_contact_motion_m"] += float(np.linalg.norm(delta[:2]))
                # Actual round-head contact earns poke progress even when the
                # two-finger force-based grasp detector flickers during impact.
                correct = m == "poke" or (held[m] and inner)
                progress = max(0.0, float(delta @ d))
                if correct:
                    out["correct_contact_steps"] += 1
                    out["correct_progress_m"] += progress
                else:
                    out["incorrect_contact_motion_m"] += float(np.linalg.norm(delta[:2]))
                if hook_near_target and correct:
                    out["near_target_inner_progress_m"] += progress
        if "scoop" in self.handheld:
            pan = self.handheld["scoop"]
            s["pan_touched_target"] |= pan_touches_target(
                pos(pan["head"]), rotation(pan["head"]),
                pan["params"],
                l.target, TARGET_RADIUS,
            )
            if s["pan_touched_target"]:
                self.strict_event("pan_touched_target")
            out = s["handheld"]["scoop"]
            self._pan_unload.update(
                dt=dt,
                carried=(out["longest_supported_s"] >= 0.15
                         and out["max_supported_carry_m"] > 0.06
                         and out["max_supported_target_progress_m"] > 0.06),
                held=held["scoop"],
                contact=bool({"scoop_head", "scoop_shaft", "scoop_handle"} & set(current)),
                head_contact="scoop_head" in current,
                table_contact=table_contact, head=pos(pan["head"]),
                rotation=rotation(pan["head"]), cube=cube, target=l.target,
            )
            for key in ("started", "tilted_release", "post_release_contact"):
                if s["pan_unload"][key]:
                    self.strict_event("pan_unload_" + key)
        if self.ball is not None:
            self._monitor_ball(
                ball_held, dt, self.ball._bodies[0].entity in robot_touches
            )
        if self.cue is not None:
            self._monitor_cue(cue_held)

    def _hook_underside_only(self):
        head = self.handheld["hook"]["head"]
        entities = {self.cube._bodies[0].entity, head._bodies[0].entity}
        r = rotation(head)
        normals = []
        for c in self.scene.px.get_contacts():
            if {b.entity for b in c.bodies} == entities:
                sign = 1 if c.bodies[0].entity == self.cube._bodies[0].entity else -1
                normals.extend(r.T @ (sign * np.asarray(p.normal)) for p in c.points
                               if np.linalg.norm(p.impulse) > 1e-7)
        return bool(normals) and all(underside_normal(n) for n in normals)

    def _hook_inner_only(self):
        head = self.handheld["hook"]["head"]
        entities = {self.cube._bodies[0].entity, head._bodies[0].entity}
        r, origin = rotation(head), pos(head)
        scale = self.handheld["hook"]["params"].get("hook_head_scale", 1.0)
        faces = []
        for c in self.scene.px.get_contacts():
            if {b.entity for b in c.bodies} != entities:
                continue
            sign = 1 if c.bodies[0].entity == self.cube._bodies[0].entity else -1
            for p in c.points:
                if np.linalg.norm(p.impulse) > 1e-7:
                    faces.append(hook_inner_contact(r.T @ (np.asarray(p.position) - origin),
                                                   r.T @ (sign * np.asarray(p.normal)), scale))
        return bool(faces) and all(faces)

    def _strict_pan_regions(self):
        from .pan import classify_pan_contact

        cube_entity = self.cube._bodies[0].entity
        head = self.handheld["scoop"]["head"]
        head_entity = head._bodies[0].entity
        r = rotation(head)
        origin = pos(head)
        regions = set()
        for c in self.scene.px.get_contacts():
            if {b.entity for b in c.bodies} != {cube_entity, head_entity}:
                continue
            sign = 1 if c.bodies[0].entity == cube_entity else -1
            for p in c.points:
                if np.linalg.norm(p.impulse) > 1e-7:
                    regions.add(
                        classify_pan_contact(
                            r.T @ (np.asarray(p.position) - origin),
                            r.T @ (sign * np.asarray(p.normal)),
                            self.handheld["scoop"]["params"],
                        )
                    )
        return regions

    def _monitor_ball(self, held, dt, robot_contact):
        e = self.strict["ball"]
        l = self.layout
        p = pos(self.ball)
        body = self.ball._bodies[0]
        e["grasped"] |= held
        if robot_contact or held:
            self._strict_ball_last_robot_contact = self.time
            if self._strict_ball_separated:
                # Touching the rolling ball is diagnostic, not a violation.
                e["robot_recontact_after_release"] = True
            if held and (self._strict_ball_separated or self._strict_ball_hit):
                # Picking the ball up again starts a new release. Do not reuse
                # an earlier roll/impact to qualify a later placement or throw.
                # Brief finger brushes alone retain the roll, while the latest
                # robot-contact time still gates robot-free cube impact.
                self._strict_ball_separated = False
                self._strict_ball_release = None
                self._strict_ball_hit = False
                self._strict_ball_ground_start = None
                self.first_contact.pop("ball", None)
                e.update(
                    released_on_ramp=False,
                    ramp_after_release_s=0.0,
                    descent_after_release_m=0.0,
                    rolling_rotation_rad=0.0,
                    flat_runout_m=0.0,
                    free_before_hit_s=0.0,
                    free_hit=False,
                    impact_speed_mps=0.0,
                )
            self._strict_ball_separation_candidate = None
        elif e["grasped"] and not self._strict_ball_separated:
            if self._strict_ball_separation_candidate is None:
                # A grasp-detector dropout while a finger still touches the
                # ball is not a release. Start at the first contact-free step,
                # retaining its speed so deceleration cannot hide a throw.
                self._strict_ball_separation_candidate = (
                    self.time,
                    p.copy(),
                    float(np.linalg.norm(body.linear_velocity)),
                    float(np.linalg.norm(body.angular_velocity)) * l.ball_radius,
                )
                e.update(
                    released_on_ramp=False,
                    ramp_after_release_s=0.0,
                    descent_after_release_m=0.0,
                    rolling_rotation_rad=0.0,
                    flat_runout_m=0.0,
                )
                self._strict_ball_ground_start = None
            t0, p0, speed0, spin0 = self._strict_ball_separation_candidate
            if self.time - t0 >= 0.02 - 1e-9:
                self._strict_ball_separated = True
                self._strict_ball_release = (t0, p0.copy())
                local = p0 - l.toe
                fraction = float(-local @ l.ramp_direction / l.ramp_run)
                normal = l.ramp_direction * np.sin(l.ramp_angle) + [
                    0,
                    0,
                    np.cos(l.ramp_angle),
                ]
                e["released_on_ramp"] = bool(
                    0.0 < fraction < 1.05
                    and abs(local @ l.across)
                    < l.ramp_width_m / 2 - l.ball_radius + 0.003
                    and abs(local @ normal - l.ball_radius) < 0.008
                )
                e["launched"] |= speed0 > 0.12 or spin0 > 0.12
                self.strict_event(
                    "ball_release",
                    on_ramp=e["released_on_ramp"],
                    launch_speed_mps=speed0,
                    release_time=t0,
                )
                self.strict_event(
                    "ball_fully_released",
                    launch_speed_mps=speed0,
                    spin_surface_speed_mps=spin0,
                )
        # Include the confirmation interval in rolling evidence. A candidate
        # interrupted by contact is discarded and the next one starts fresh.
        release = self._strict_ball_release or self._strict_ball_separation_candidate
        if release is not None and not held and not robot_contact and not self._strict_ball_hit:
            start = release[1]
            ramp_contact = any(
                {b.entity for b in c.bodies}
                == {body.entity, self.ramp._bodies[0].entity}
                and any(np.linalg.norm(p.impulse) > 1e-7 for p in c.points)
                for c in self.scene.px.get_contacts()
            )
            if ramp_contact and not robot_contact:
                e["ramp_after_release_s"] += dt
                e["rolling_rotation_rad"] += (
                    float(np.linalg.norm(body.angular_velocity)) * dt
                )
            e["descent_after_release_m"] = max(
                e["descent_after_release_m"], float(start[2] - p[2])
            )
            if (p - l.toe) @ l.ramp_direction > l.ball_radius and abs(
                p[2] - l.ball_radius
            ) < 0.004:
                if self._strict_ball_ground_start is None:
                    self._strict_ball_ground_start = p.copy()
                e["flat_runout_m"] = max(
                    e["flat_runout_m"],
                    float((p - self._strict_ball_ground_start) @ l.ramp_direction),
                )
        if "ball" in self.current_contacts:
            e["held_hit"] |= held or self.events["ball_held_hit"]
            e["free_hit"] |= not held
            if not self._strict_ball_hit:
                self._strict_ball_hit = True
                e["impact_speed_mps"] = self.impact_speed["ball"]
                if self._strict_ball_release is not None:
                    e["free_before_hit_s"] = self.time - max(
                        self._strict_ball_release[0],
                        self._strict_ball_last_robot_contact,
                    )
                self.strict_event("ball_impact", held=held)

    def _monitor_cue(self, held):
        e = self.strict["cue"]
        l = self.layout
        q = self.cue.get_qpos()[0].cpu().numpy()
        v = self.cue.get_qvel()[0].cpu().numpy()
        e["limits_respected"] &= bool(-0.002 <= q[1] <= l.cue_stroke_m + 0.002 and abs(q[0]) <= 4 * np.pi + np.radians(2))
        if "cue" in self.current_contacts:
            cube_entity = self.cube._bodies[0].entity
            cue_entity = self.cue_link._bodies[0].entity
            r = rotation(self.cue_link)
            contact_points = []
            for c in self.scene.px.get_contacts():
                if {b.entity for b in c.bodies} != {cube_entity, cue_entity}:
                    continue
                for p in c.points:
                    if np.linalg.norm(p.impulse) <= 1e-7:
                        continue
                    contact_points.append(np.asarray(p.position))
                    local = r.T @ (np.asarray(p.position) - pos(self.cue_link))
                    if (
                        abs(local[0] - l.cue_front_m) > 0.011
                        or np.linalg.norm(local[1:]) > 0.011
                    ):
                        e["non_tip_contact"] = True
            if self.time - self._strict_cue_last_hit > 0.03:
                # Classify the incoming motion, before collision impulses can
                # rotate the passive bearing. The previous substep is 2 ms old.
                incoming = self._strict_previous_cue_velocity
                contact_speed = max((np.linalg.norm(cue_point_velocity(
                    incoming[1], incoming[0], r[:, 0], point - l.cue_pivot))
                    for point in contact_points), default=0.0)
                e["impact_contact_speed_mps"] = max(e["impact_contact_speed_mps"], float(contact_speed))
                e["dynamic_hit"] |= bool(contact_speed > 0.08)
                e["impact_forward_speed_mps"] = (
                    float(incoming[1])
                    if self._strict_cue_strikes == 0
                    else min(e["impact_forward_speed_mps"], float(incoming[1]))
                )
                axial, rotational, swing = cue_impact_components(
                    float(incoming[1]),
                    float(incoming[0]),
                    l.cue_front_m + float(q[1]),
                    float(q[0]),
                    np.radians(l.target_heading_deg),
                )
                e["swing_hit"] |= swing
                legal = bool(held and incoming[1] > 0 and not e["non_tip_contact"])
                e["held_tip_hit"] = (
                    legal
                    if self._strict_cue_strikes == 0
                    else e["held_tip_hit"] and legal
                )
                self._strict_cue_strikes += 1
                self.strict_event(
                    "cue_impact",
                    held=held,
                    forward_speed_mps=float(incoming[1]),
                    angular_speed_radps=float(incoming[0]),
                    contact_speed_mps=float(contact_speed),
                    target_directed_axial_speed_mps=axial,
                    target_directed_rotational_speed_mps=rotational,
                )
            self._strict_cue_last_hit = self.time
        last = self.last_contact.get("cue")
        if last:
            e["coast_after_last_contact_m"] = float(
                (pos(self.cube) - last["xyz"]) @ unit(l.target_heading_deg)
            )
        self._strict_previous_cue_velocity = v.copy()

    def method_success(self, method):
        if not hasattr(self, "strict"):
            return False
        return verdict(method, self.available_methods, self.strict)["success"]

    def result(self):
        result = super().result()
        result.update(
            criteria_version=VERSION,
            criteria=verdict(
                self._required_method, self.available_methods, self.strict
            ),
            strict_evidence=copy.deepcopy(self.strict),
            strict_events=copy.deepcopy(self.strict_events),
        )
        return result
