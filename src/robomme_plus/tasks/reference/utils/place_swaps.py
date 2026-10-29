"""Clock-driven elliptical target swaps, gated only at the start of each round."""

import numpy as np

from robomme_plus.common import sim as base
from robomme_plus.common.sim import position
from robomme_plus.tasks.reference.utils.place_corrections import (
    FINAL_CORRECTION_ROUNDS,
    GOAL_LOCATION_TOLERANCE_M,
    plan_correction_pair,
)
from robomme_plus.tasks.reference.utils.place_visibility import MAX_ARM_OCCLUSION, SwapVisibility

HEIGHT_M = 0.10
TICK_STEPS = 1
SWAP_STEPS = 40
ELLIPSE_RADIUS_MULTIPLIER = 4.0
LAYOUT_LOW = np.array([-0.33, -0.32])
LAYOUT_HIGH = np.array([0.12, 0.32])
PARTNER_NEAREST_K = 3
SINGLE_PAIR_PROBABILITY = 0.70
EXECUTION_MAX_ROUNDS = 0


def sample_random_layout(count, radius, rng, obstacles=()):
    """Uniform position proposals, constrained only by physical separation."""
    for _ in range(120):
        points = []
        for _ in range(count):
            for _ in range(10000):
                point = rng.uniform(LAYOUT_LOW, LAYOUT_HIGH)
                if all(
                    np.linalg.norm(point - other) > 2 * radius + 0.014
                    for other in points
                ) and all(
                    np.linalg.norm(point - other) > radius + r + 0.014
                    for other, r in obstacles
                ):
                    points.append(point)
                    break
            else:
                break
        if len(points) == count:
            rng.shuffle(points)
            return np.asarray(points)
    raise RuntimeError("Cannot fit target rings around the native objects")


def sample_pair_count(rng):
    return 1 if rng.random() < SINGLE_PAIR_PROBABILITY else 2


def plan_neighbor_pairs(points, rng, pinned=(), maximum=2, *, radius=0.04):
    """Pair nearest-three neighbors and exchange them on opposite half-ellipses."""
    points = np.asarray(points, dtype=float)
    available = [i for i in range(len(points)) if i not in pinned]
    pairs, selections = [], []
    while len(available) >= 2 and len(pairs) < min(maximum, 2):
        anchor = int(rng.choice(available))
        candidates = [i for i in available if i != anchor]
        candidates.sort(
            key=lambda i: (float(np.linalg.norm(points[i] - points[anchor])), i)
        )
        nearest = candidates[:PARTNER_NEAREST_K]
        partner = int(rng.choice(nearest))
        selections.append(
            {
                "anchor_index": anchor,
                "nearest_indices": nearest,
                "partner_index": partner,
                "partner_rank": nearest.index(partner) + 1,
                "distance_m": float(np.linalg.norm(points[partner] - points[anchor])),
            }
        )
        pairs.append((anchor, partner))
        available.remove(anchor)
        available.remove(partner)
    return pair_paths(points, pairs, radius), pairs, selections


def pair_paths(points, pairs, radius):
    """Use identical elliptical motion for ordinary and final correction pairs."""
    points = np.asarray(points, dtype=float)
    u = np.linspace(0, 1, SWAP_STEPS + 1)
    angle = (np.pi * u * u * (3 - 2 * u))[:, None]
    paths = np.repeat(points[None], len(u), axis=0)
    for anchor, partner in pairs:
        delta = points[partner] - points[anchor]
        distance = np.linalg.norm(delta)
        if distance > 1e-9:
            direction = delta / distance
            normal = np.array([-direction[1], direction[0]])
            center = (points[anchor] + points[partner]) / 2
            # The endpoint axis is half the pair separation. The transverse
            # semi-axis is four target radii, even for closely spaced pairs.
            offset = (
                -0.5 * distance * np.cos(angle) * direction
                + ELLIPSE_RADIUS_MULTIPLIER * radius * np.sin(angle) * normal
            )
            paths[:, anchor] = center + offset
            paths[:, partner] = center - offset
        paths[0, anchor], paths[0, partner] = points[anchor], points[partner]
        paths[-1, anchor], paths[-1, partner] = points[partner], points[anchor]
    return paths


class ClockedSwaps:
    """Finish video rounds before handoff; execution never moves targets."""

    def __init__(self, env, actors, radius, speed, steps=0, obstacles=(), **kwargs):
        if not 4 <= len(actors) <= 8:
            raise ValueError("Clocked VideoPlace supports four to eight target rings")
        self.env, self.actors, self.radius = env, actors, radius
        self.visibility = SwapVisibility(env, radius)
        self.z = [float(position(actor, initial=False)[2]) for actor in actors]
        self.q = [
            actor.pose.q.detach().cpu().numpy().reshape(-1, 4)[0].copy()
            for actor in actors
        ]
        self.seed = int(env.seed) + 872431
        layout_rng, swap_rng, count_rng = np.random.SeedSequence(self.seed).spawn(3)
        self.rng = np.random.default_rng(swap_rng)
        self.count_rng = np.random.default_rng(count_rng)
        points = np.asarray(env._video_place_layout, dtype=float).copy()
        if points.shape != (len(actors), 2):
            raise ValueError("Initial layout must match all target actors")
        for i, actor in enumerate(actors):
            base.set_position(actor, [*points[i], self.z[i]], self.q[i])
        self.paths = points[None].copy()
        self.pairs = []
        self.history = [points.copy()]
        self.clock = self.cursor = self.phase = 0
        self.active = False
        self.execution_start = None
        self.stage = "demonstration"
        self.demonstration_closing = False
        self.goal_index = None
        self.reference_xy = None
        self.correction_active = False
        self.correction_rounds_started = 0
        self.correction_rounds_completed = 0
        self.correction_rng = np.random.default_rng(self.seed + 193381)
        self.execution_round_budget = 0
        self.execution_rounds_started = 0
        self.execution_rounds_completed = 0
        self.pinned = set()
        self.counts = np.zeros(len(actors), dtype=int)
        self._last_gate = None
        self.metadata = {
            "strategy": "ordinary swaps start above 10 cm on the robot side; finish active round during final cube return; move to central air then two one-pair correction swaps; execution targets fixed",
            "seed": self.seed,
            "actor_count": len(actors),
            "radius_m": radius,
            "actor_names": [actor.name for actor in actors],
            "height_threshold_m": HEIGHT_M,
            "ordinary_start_condition": "tcp_z > 0.10 m and tcp_x < 0 m",
            "ordinary_visibility_gate": {
                "camera": "base_camera",
                "maximum_arm_occluded_fraction": MAX_ARM_OCCLUSION,
                "policy": "skip a pair if either target is more than half covered by robot links",
                "area": "full projected target cylinder, including pixels hidden by the arm",
                "scope": "ordinary round starts only",
            },
            "visibility_rejections": [],
            "tick_control_steps": TICK_STEPS,
            "round_control_steps": SWAP_STEPS,
            "max_pairs_per_round": 2,
            "clock_steps": 0,
            "ticks": [],
            "rounds": [],
            "gate_changes": [],
            "occupied_pin_events": [],
            "aborted_rounds": [],
            "movement_steps": 0,
            "simultaneously_moving_max": 0,
            "completed_rounds": 0,
            "low_height_movement_steps": 0,
            "pair_count_probabilities": {
                "1": SINGLE_PAIR_PROBABILITY,
                "2": 1 - SINGLE_PAIR_PROBABILITY,
            },
            "demonstration_end": {
                "new_rounds_disabled_clock": None,
                "last_round_finished_clock": None,
            },
            "execution": {
                "max_rounds": EXECUTION_MAX_ROUNDS,
                "round_budget": 0,
                "rounds_started": 0,
                "rounds_completed": 0,
                "carryover_rounds": 0,
                "carryover_counts_toward_budget": False,
            },
            "max_speed_m_per_control_step": 0.0,
            "cube_poses_written_by_challenge": 0,
            "demonstration_swap_wait_steps": 0,
            "minimum_moving_tcp_height_m": None,
            "layout_sampling": "uniform position proposals with clearance only; no swap-feasibility rejection",
            "layout_bounds_m": [LAYOUT_LOW.tolist(), LAYOUT_HIGH.tolist()],
            "initial_target_xy": points.tolist(),
            "pair_selection": "random anchor; random partner among its three nearest unused targets; repeat for second pair",
            "partner_nearest_k": PARTNER_NEAREST_K,
            "swap_path": "opposite half-ellipses with smooth angular progress",
            "ellipse_radius_multiplier": ELLIPSE_RADIUS_MULTIPLIER,
            "ellipse_transverse_radius_m": ELLIPSE_RADIUS_MULTIPLIER * radius,
            "target_collision_avoidance": False,
            "placement_policy": "above-target approach and tracked descent; no threshold waypoint",
        }

    def enable_final_corrections(self, goal_index):
        self.goal_index = goal_index
        self.metadata["final_corrections"] = {
            "required_rounds": FINAL_CORRECTION_ROUNDS,
            "pairs_per_round": 1,
            "reference_target_name": self.actors[goal_index].name,
            "reference_xy": None,
            "reference_step": None,
            "location_tolerance_m": GOAL_LOCATION_TOLERANCE_M,
            "rounds_started": 0,
            "rounds_completed": 0,
            "hold_start_clock": None,
            "hold_end_clock": None,
        }

    def record_reference(self, xy, step):
        if self.reference_xy is not None:
            raise RuntimeError("The requested goal placement was recorded twice")
        self.reference_xy = np.asarray(xy, dtype=float).copy()
        self.metadata["final_corrections"].update(
            reference_xy=self.reference_xy.tolist(), reference_step=step
        )

    @property
    def corrections_complete(self):
        return self.correction_rounds_completed == FINAL_CORRECTION_ROUNDS

    def begin_final_corrections(self):
        if not self.demonstration_closing or self.pairs:
            raise RuntimeError("Finish ordinary video swaps before final corrections")
        if self.reference_xy is None:
            raise RuntimeError("The requested goal placement was not observed")
        if self.correction_active or self.correction_rounds_started:
            raise RuntimeError("Final correction swaps were already started")
        self.correction_active = True
        self.metadata["final_corrections"].update(
            hold_start_clock=self.clock,
            hold_tcp_position=position(self.env.agent.tcp, initial=False).tolist(),
        )

    @property
    def start_enabled(self):
        return (
            self.active
            and position(self.env.agent.tcp, initial=False)[0] < 0
            and self.stage == "demonstration"
            and not self.demonstration_closing
        )

    def finish_demonstration(self):
        """Let the current round finish during normal cube/arm return motion."""
        if self.demonstration_closing:
            return
        self.demonstration_closing = True
        self.metadata["demonstration_end"].update(
            new_rounds_disabled_clock=self.clock,
            remaining_round_steps=SWAP_STEPS - self.phase if self.pairs else 0,
        )
        if not self.pairs:
            self.metadata["demonstration_end"]["last_round_finished_clock"] = self.clock

    def begin_execution(self):
        """Require a finished video round; never continue or start execution swaps."""
        if self.stage == "execution":
            return
        if self.pairs:
            raise RuntimeError(
                "Video ended with an unfinished swap; finish it in the recorded demonstration"
            )
        if self.goal_index is not None:
            if not self.corrections_complete:
                raise RuntimeError("Video ended before both correction swaps completed")
            final_xy = position(self.actors[self.goal_index], initial=False)[:2]
            if np.linalg.norm(final_xy - self.reference_xy) <= GOAL_LOCATION_TOLERANCE_M:
                raise RuntimeError("Execution goal returned to its demonstrated position")
        self.stage = "execution"
        self.execution_start = base.step_number(self.env)
        self.metadata["execution"].update(
            start_clock=self.clock,
            target_xy=[
                position(actor, initial=False)[:2].tolist() for actor in self.actors
            ],
        )

    def update(self):
        self.clock += 1
        self.metadata["clock_steps"] = self.clock
        if not getattr(self.env, "demonstration_record_traj", True):
            self.begin_execution()
        before = np.array([position(actor, initial=False)[:2] for actor in self.actors])
        tcp = position(self.env.agent.tcp, initial=False)
        tcp_x, tcp_z = float(tcp[0]), float(tcp[2])
        self.active = tcp_z > HEIGHT_M
        ordinary_gate = self.active and tcp_x < 0
        if ordinary_gate != self._last_gate:
            self.metadata["gate_changes"].append(
                {
                    "clock": self.clock,
                    "step": base.step_number(self.env),
                    "active": ordinary_gate,
                    "tcp_x_m": tcp_x,
                    "tcp_height_m": tcp_z,
                }
            )
            self._last_gate = ordinary_gate
        if self.stage == "execution":
            return
        pinned = set()
        moving_indices = {i for pair in self.pairs for i in pair}
        for cube in self.env.all_cubes:
            p = position(cube, initial=False)
            if p[2] < 0.055:
                distances = np.linalg.norm(before - p[:2], axis=1)
                nearby = np.flatnonzero(distances <= 0.055)
                # Active exchanges always finish, including beneath low cubes.
                # Only rings outside the current exchange can become occupied.
                eligible = [i for i in nearby if i not in moving_indices]
                # Crossing arcs can bring several rings beneath one cube. The
                # cube occupies only its closest ring, not every nearby ring.
                if eligible:
                    pinned.add(min(eligible, key=lambda i: (distances[i], i)))
        if pinned != self.pinned:
            self.metadata["occupied_pin_events"].append(
                {
                    "clock": self.clock,
                    "target_names": [self.actors[i].name for i in sorted(pinned)],
                }
            )
        self.pinned = pinned
        if self.clock % TICK_STEPS == 0:
            tick = {
                "clock": self.clock,
                "tcp_x_m": tcp_x,
                "tcp_height_m": tcp_z,
                "started_pairs": 0,
                "reason": "below_height" if not self.active else "round_in_progress",
            }
            correction_due = (
                self.correction_active and self.active
                and self.correction_rounds_started < FINAL_CORRECTION_ROUNDS
            )
            if (self.start_enabled or correction_due) and not self.pairs:
                target_visibility, rejected = {}, []
                if correction_due:
                    requested = 1
                    pair, selection = plan_correction_pair(
                        before, self.goal_index, self.reference_xy,
                        self.correction_rng, pinned,
                    )
                    pairs, selections = [pair], [selection]
                    paths = pair_paths(before, pairs, self.radius)
                    self.correction_rounds_started += 1
                    self.metadata["final_corrections"]["rounds_started"] = self.correction_rounds_started
                else:
                    requested = sample_pair_count(self.count_rng)
                    paths, pairs, selections = plan_neighbor_pairs(
                        before, self.rng, pinned, maximum=requested, radius=self.radius
                    )
                    if pairs:
                        candidate_indices = sorted({i for pair in pairs for i in pair})
                        measurements = dict(zip(candidate_indices, self.visibility.measure(
                            [self.actors[i] for i in candidate_indices]
                        )))
                        accepted = []
                        for pair, selection in zip(pairs, selections):
                            if all(measurements[i]["allowed"] for i in pair):
                                accepted.append((pair, selection))
                            else:
                                rejected.append(pair)
                        if rejected:
                            self.metadata["visibility_rejections"].append({
                                "clock": self.clock,
                                "pairs": [[self.actors[i].name for i in pair] for pair in rejected],
                                "targets": {
                                    self.actors[i].name: measurements[i]
                                    for pair in rejected for i in pair
                                },
                            })
                        pairs = [pair for pair, _ in accepted]
                        selections = [selection for _, selection in accepted]
                        paths = pair_paths(before, pairs, self.radius)
                        target_visibility = {
                            self.actors[i].name: measurements[i] for pair in pairs for i in pair
                        }
                if pairs:
                    self.paths, self.pairs, self.phase = paths, pairs, 0
                    for i, j in pairs:
                        self.counts[[i, j]] += 1
                    self.metadata["target_swap_counts"] = self.counts.tolist()
                    self.metadata["rounds"].append(
                        {
                            "clock": self.clock,
                            "stage": self.stage,
                            "kind": "correction" if correction_due else "ordinary",
                            "start_tcp_position_m": tcp.tolist(),
                            **({"start_target_visibility": target_visibility} if not correction_due else {}),
                            "start_tcp_height_m": tcp_z,
                            "requested_pairs": requested,
                            "selections": selections,
                            "pairs": [
                                [self.actors[i].name, self.actors[j].name]
                                for i, j in pairs
                            ],
                            "pair_distances_m": [
                                float(np.linalg.norm(before[i] - before[j]))
                                for i, j in pairs
                            ],
                            "start_xy": paths[0].tolist(),
                            "end_xy": paths[-1].tolist(),
                        }
                    )
                tick.update(
                    requested_pairs=requested,
                    started_pairs=len(self.pairs),
                    reason="started" if self.pairs else "arm_occluded" if rejected else "not_enough_available_targets",
                )
            if (
                tick["started_pairs"]
                or tick.get("reason") in {"not_enough_available_targets", "arm_occluded"}
            ):
                self.metadata["ticks"].append(tick)
        if not self.pairs:
            return
        after = self.paths[self.phase + 1]
        movement = np.linalg.norm(after - before, axis=1)
        for i, actor in enumerate(self.actors):
            if movement[i] > 1e-10:
                assert i not in pinned
                base.set_position(actor, [*after[i], self.z[i]], self.q[i])
        self.phase += 1
        self.cursor += 1
        self.history.append(after.copy())
        self.metadata["movement_steps"] += 1
        self.metadata["low_height_movement_steps"] += int(not self.active)
        previous_height = self.metadata["minimum_moving_tcp_height_m"]
        self.metadata["minimum_moving_tcp_height_m"] = min(
            tcp_z, previous_height if previous_height is not None else tcp_z
        )
        self.metadata["simultaneously_moving_max"] = max(
            self.metadata["simultaneously_moving_max"], int((movement > 1e-6).sum())
        )
        self.metadata["max_speed_m_per_control_step"] = max(
            self.metadata["max_speed_m_per_control_step"], float(movement.max())
        )
        if self.phase == SWAP_STEPS:
            self.metadata["completed_rounds"] += 1
            self.metadata["rounds"][-1].update(
                completed_clock=self.clock, end_tcp_height_m=tcp_z
            )
            if self.metadata["rounds"][-1]["kind"] == "correction":
                final_xy = after[self.goal_index]
                if np.linalg.norm(final_xy - self.reference_xy) <= GOAL_LOCATION_TOLERANCE_M:
                    raise RuntimeError("A correction swap restored the demonstrated goal position")
                self.correction_rounds_completed += 1
                self.metadata["final_corrections"].update(
                    rounds_completed=self.correction_rounds_completed,
                    final_goal_xy=final_xy.tolist(),
                    goal_displacement_m=float(np.linalg.norm(final_xy - self.reference_xy)),
                )
                if self.corrections_complete:
                    self.correction_active = False
                    self.metadata["final_corrections"]["hold_end_clock"] = self.clock
            if self.demonstration_closing:
                self.metadata["demonstration_end"]["last_round_finished_clock"] = (
                    self.clock
                )
            self.pairs = []
            self.phase = 0


def clocked_put(env, planner, target=None, height=None):
    """Approach above the ring, track it during descent, release, and lift.

    Every descent plan ends at placement height. Tracking refreshes the goal
    while the ring moves; it never inserts a waypoint to trip the height gate.
    Only ClockedSwaps.update decides whether targets move.
    """
    import sapien

    native = env.unwrapped
    q = native.agent.tcp.pose.q.detach().cpu().numpy().reshape(-1, 4)[0]
    tcp = position(target, initial=False).copy()
    tcp[2] = 0.15
    if planner.move_to_pose_with_screw(sapien.Pose(tcp, q)) == -1:
        return -1
    descent_steps = 0
    while descent_steps < 240:
        tcp[:2] = position(target, initial=False)[:2]
        tcp[2] = height if height is not None else 0.015
        plan = planner.move_to_pose_with_screw(sapien.Pose(tcp, q), dry_run=True)
        if not isinstance(plan, dict) or plan.get("status") != "Success":
            return -1
        for start in range(0, len(plan["position"]), 8):
            end = min(start + 8, len(plan["position"]))
            if descent_steps + end - start > 240:
                return -1
            planner.follow_path(
                {
                    **plan,
                    "position": plan["position"][start:end],
                    "velocity": plan["velocity"][start:end],
                }
            )
            descent_steps += end - start
            if np.linalg.norm(position(target, initial=False)[:2] - tcp[:2]) > 0.006:
                break
        else:
            routes = getattr(native, "_limit_routes", None)
            if routes is not None and any(
                target is routes.actors[i] for pair in routes.pairs for i in pair
            ):
                continue
            break
    else:
        return -1
    planner.open_gripper(t=12)
    tcp = position(native.agent.tcp, initial=False).copy()
    tcp[2] = 0.15
    return planner.move_to_pose_with_screw(sapien.Pose(tcp, q))


class LevelSwaps(ClockedSwaps):
    """Keep swap speed unchanged; easy doubles the minimum start spacing."""

    def __init__(self, env, *args, **kwargs):
        self.start_interval = env._limit_config["swap_start_interval_steps"]
        if self.start_interval < SWAP_STEPS:
            raise ValueError("Swap start interval cannot be shorter than a round")
        super().__init__(env, *args, **kwargs)
        self.metadata.update(
            level=env._limit_config["level"],
            swap_start_interval_steps=self.start_interval,
            swap_frequency_policy="minimum start-to-start interval; clock continues outside ordinary start region",
        )

    @property
    def start_enabled(self):
        rounds = self.metadata["rounds"]
        due = not rounds or self.clock >= rounds[-1]["clock"] + self.start_interval
        return super().start_enabled and due
