"""Recurring, collision-checked local rearrangements during PickHighlight."""

from itertools import combinations, permutations

import numpy as np

from robomme_plus.tasks.reference.utils.highlight_levels import DENSE_CUBE_BOUNDS

DURATION = 40
INTERVAL_PER_LEVEL = {"easy": 100, "medium": 60, "hard": 40}
CLUSTERS_PER_LEVEL = {"easy": (1, 2), "medium": (1, 3), "hard": (1, 3)}
CLUSTER_SIZE_PER_LEVEL = {"easy": (2, 3), "medium": (2, 4), "hard": (2, 5)}
START_CLEARANCE = .09
STOP_CLEARANCE = .06
CLUSTER_DIAMETER = .30
MOTION_GAP = .008
MIN_DISPLACEMENT = .10
MAX_EXCURSION = .32


def _segment_distances(before, after):
    velocity = after - before
    t = np.clip(-np.sum(before * velocity, axis=-1)
                / np.maximum(np.sum(velocity * velocity, axis=-1), 1e-20), 0, 1)
    return np.linalg.norm(before + t[..., None] * velocity, axis=-1)


def clear_path(path, stationary, button, tcp, radius, *, gripper_clearance=START_CLEARANCE,
               endpoint_gap=None, moving_obstacles=None):
    """Check whole linear frame intervals, including changing pair distances."""
    path = np.asarray(path)
    bounds = np.asarray(DENSE_CUBE_BOUNDS)
    if (np.any(path < bounds[:, 0] + radius + .002)
            or np.any(path > bounds[:, 1] - radius - .002)):
        return False
    if _segment_distances(path[:-1] - tcp, path[1:] - tcp).min() < gripper_clearance:
        return False
    if _segment_distances(path[:-1] - button, path[1:] - button).min() < radius + np.sqrt(2)*.0375 + MOTION_GAP:
        return False
    a, b = np.triu_indices(path.shape[1], 1)
    if len(a):
        relative = path[:, a] - path[:, b]
        if _segment_distances(relative[:-1], relative[1:]).min() < 2*radius + MOTION_GAP:
            return False
        if endpoint_gap is not None and np.linalg.norm(relative[-1], axis=-1).min() < 2*radius + endpoint_gap:
            return False
    stationary = np.asarray(stationary).reshape(-1, 2)
    if len(stationary):
        relative = path[:, :, None] - stationary
        if _segment_distances(relative[:-1], relative[1:]).min() < 2*radius + MOTION_GAP:
            return False
        if endpoint_gap is not None and np.linalg.norm(relative[-1], axis=-1).min() < 2*radius + endpoint_gap:
            return False
    if moving_obstacles is not None and np.asarray(moving_obstacles).size:
        relative = path[:, :, None] - np.asarray(moving_obstacles)[:, None]
        if _segment_distances(relative[:-1], relative[1:]).min() < 2*radius + MOTION_GAP:
            return False
        if endpoint_gap is not None and np.linalg.norm(relative[-1], axis=-1).min() < 2*radius + endpoint_gap:
            return False
    return True


def changed_arrangement(start, goal):
    """Reject small drifts and translations that preserve relative positions."""
    if np.linalg.norm(goal - start, axis=-1).min() < MIN_DISPLACEMENT:
        return False
    before = start[:, None] - start[None, :]
    after = goal[:, None] - goal[None, :]
    # Every cube changes sides relative to another cube by at least 90 degrees.
    reversed_pairs = np.sum(before * after, axis=-1) < 0
    if not reversed_pairs.any(axis=1).all():
        return False
    a, b = np.triu_indices(len(start), 1)
    return reversed_pairs[a, b].mean() >= .5


def sample_destinations(points, rng, assignments):
    """New positions near shuffled local sites, without exact slot matching."""
    assignment = assignments[int(rng.integers(len(assignments)))]
    angle = rng.uniform(-np.pi, np.pi, len(points))
    offset = rng.uniform(.01, .025, (len(points), 1))
    return points[assignment] + offset * np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def group_path(points, goal, rng, *, arc=False):
    """Plan smooth paths to already chosen destinations; never weaken the goal."""
    t = np.linspace(0, 1, DURATION + 1)[:, None, None]
    # Different progress rates help cubes pass one another without moving rigidly.
    t = t ** rng.uniform(.8, 1.25, (1, len(points), 1))
    ease = t*t*(3-2*t)
    if arc:
        center = (points.mean(axis=0) + goal.mean(axis=0)) / 2
        before, after = points - center, goal - center
        angles = np.arctan2(before[:, 1], before[:, 0])
        final_angles = np.arctan2(after[:, 1], after[:, 0])
        direction = int(rng.choice([-1, 1]))
        turn = direction * np.mod(direction * (final_angles - angles), 2*np.pi)
        theta = angles[None, :, None] + ease * turn[None, :, None]
        radii, final_radii = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
        distance = radii[None, :, None] + ease * (final_radii-radii)[None, :, None]
        path = center + distance * np.concatenate((np.cos(theta), np.sin(theta)), axis=-1)
    else:
        delta = goal-points
        normal = np.stack((-delta[:, 1], delta[:, 0]), axis=-1)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
        bend = normal * rng.uniform(-.13, .13, (len(points), 1))
        path = points + ease * delta + np.sin(np.pi*ease) * bend
    path[0], path[-1] = points, goal
    return path


def spatial_cell(point):
    """Nine equal regions across the cube pool, including its center."""
    bounds = np.asarray(DENSE_CUBE_BOUNDS)
    cell = np.clip(((point - bounds[:, 0]) / np.diff(bounds).ravel() * 3).astype(int), 0, 2)
    return tuple(int(v) for v in cell)


def candidate_groups(positions, far, rng, max_size, cube_waits):
    """Favor waiting cubes, with weight 1 + batches skipped since movement.

    Anchors are sampled across all eligible cubes without coordinate preference.
    Partner combinations use the product of their members' weights, so each
    member's waiting time increases its chance. Safety still constrains selection.
    """
    weights = {i: 1 + cube_waits.get(i, 0) for i in far}
    # Exponential races give a weighted random order without replacement.
    anchors = sorted(far, key=lambda i: rng.exponential() / weights[i])
    seen = set()
    for anchor in anchors:
        neighbors = [i for i in far if i != anchor and
                     np.linalg.norm(positions[i] - positions[anchor]) <= CLUSTER_DIAMETER]
        for size in rng.permutation(np.arange(2, min(max_size, len(neighbors)+1)+1)):
            groups = list(combinations(neighbors, int(size)-1))
            priority = rng.exponential(size=len(groups)) / np.array([
                np.prod([weights[i] for i in group], dtype=float) for group in groups
            ])
            for trial in np.argsort(priority)[:12]:
                indices = tuple(sorted((anchor, *groups[trial])))
                if indices in seen:
                    continue
                seen.add(indices)
                points = positions[list(indices)]
                if np.linalg.norm(points[:, None] - points[None, :], axis=-1).max() <= CLUSTER_DIAMETER:
                    yield list(indices), anchor, spatial_cell(positions[anchor])


def plan_adjustment(positions, eligible, resting, button, tcp, radius, rng, *,
                    reserved=None, slots=1, cube_waits=None, max_cluster_size=5):
    """Plan one group against stationary cubes and other groups' timed paths."""
    far = [int(i) for i in eligible if np.linalg.norm(positions[i] - tcp) >= START_CLEARANCE]
    # Leave at least two cubes for each remaining slot whenever possible.
    max_size = min(max_cluster_size, max(2, len(far) - 2*(min(slots, len(far)//2)-1)))
    reserved = reserved or []
    reserved_ids = {i for plan in reserved for i in plan['indices']}
    paths = []
    for plan in reserved:
        times = np.minimum(np.arange(DURATION+1) + plan['progress'], DURATION)
        paths.append(plan['path'][times])
    moving = np.concatenate(paths, axis=1) if paths else None
    for indices, anchor, cell in candidate_groups(
            positions, far, rng, max_size, cube_waits or {}):
        stationary = positions[[i for i in resting if i not in indices and i not in reserved_ids]]
        points = positions[indices]
        assignments = [list(a) for a in permutations(range(len(points)))
                       if changed_arrangement(points, points[list(a)])]
        if not assignments:
            continue
        for attempt in range(40):
            goal = sample_destinations(points, rng, assignments)
            if not changed_arrangement(points, goal):
                continue
            if not clear_path(np.stack((goal, goal)), stationary, button, tcp, radius,
                              endpoint_gap=radius,
                              moving_obstacles=None if moving is None else moving[[-1, -1]]):
                continue
            for route in range(24):
                path = group_path(points, goal, rng, arc=route % 3 == 2)
                if np.linalg.norm(path - path[0], axis=-1).max() > MAX_EXCURSION:
                    continue
                if clear_path(path, stationary, button, tcp, radius,
                              endpoint_gap=radius, moving_obstacles=moving):
                    yaw = rng.choice([-1, 1], len(indices)) * rng.uniform(np.pi/9, np.pi/3, len(indices))
                    return dict(indices=indices, path=path, kind="rearrange", yaw=yaw,
                                anchor=anchor, region=cell)
    return None


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def planned_pose(active, index, progress):
    yaw = active['yaw'][index] * progress*progress*(3-2*progress)
    c, s = np.cos(yaw/2), np.sin(yaw/2)
    w, x, y, z = active['q'][index]
    return [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]


class LocalAdjustments:
    """Repeat difficulty-dependent batches, favoring cubes waiting to move."""

    def __init__(self, seed, half_size, *, level, start_step, visibility=None):
        self.cluster_range = CLUSTERS_PER_LEVEL[level]
        self.cluster_size = CLUSTER_SIZE_PER_LEVEL[level]
        self.interval = INTERVAL_PER_LEVEL[level]
        self.rng = np.random.default_rng([int(seed), 104733])
        self.batch_rng = np.random.default_rng([int(seed), 104734])
        self.batch_id = 0
        self.radius = np.sqrt(2) * half_size
        self.half_size = half_size
        self.events = []
        self.active = []
        self.start_step = int(start_step)
        self.next_attempt = self.start_step
        self.cube_uses = {}
        self.last_moved_batch = {}
        self.visibility = visibility

    def _finish(self, active, step, reason):
        active["event"].update(
            end_step=int(step), outcome=reason,
            end_xy=active["path"][active["progress"]].tolist(),
        )
        self.active = [plan for plan in self.active if plan is not active]
        if not self.active:
            self.next_attempt = step + self.interval

    def stop(self, step):
        for active in list(self.active):
            self._finish(active, step, "episode_finished")

    def tick(self, env):
        step = int(env.elapsed_steps.item())
        if bool(env.failureflag.any()) or bool(env.successflag.any()):
            self.stop(step)
            return
        if step < self.start_step or (not self.active and step < self.next_attempt):
            return
        # The task hides expired cue disks before this hook. Also require the
        # rendered cue state to be clear, rather than trusting the clock alone.
        cache = getattr(env, "_highlight_cache", {})
        if any(cache.get(id(cube), {}).get("is_highlighted", False)
               for cube in env.target_cubes):
            return
        xyz = np.array([_array(cube.pose.p).reshape(-1, 3)[0] for cube in env.all_cubes])
        positions = xyz[:, :2]
        tcp = _array(env.agent.tcp.pose.p).reshape(-1, 3)[0, :2]
        button = _array(env.button.pose.p).reshape(-1, 3)[0, :2]
        resting = [i for i, p in enumerate(xyz) if p[2] <= 2*self.half_size + .015]
        if self.visibility is None:
            from .highlight_visibility import CubeVisibility
            self.visibility = CubeVisibility(env, self.half_size)
        self.visibility.capture()
        quaternions = [_array(cube.pose.q).reshape(-1, 4)[0] for cube in env.all_cubes]
        visible = {}

        def currently_visible(i):
            if i not in visible:
                visible[i] = self.visibility.measure(xyz[i], quaternions[i])
            return visible[i]['allowed']

        if step >= self.next_attempt and not self.active:
            self.next_attempt = step + self.interval
            self.batch_id += 1
            # A skipped batch adds one to the next selection weight, even if
            # safety checks temporarily exclude the cube. A cube moved in the
            # previous batch starts at weight 1. Planning alone never resets it.
            cube_waits = {i: self.batch_id - 1 - self.last_moved_batch.get(i, 0)
                          for i in range(len(env.all_cubes))}
            minimum, maximum = self.cluster_range
            requested_clusters = int(self.batch_rng.integers(minimum, maximum+1))
            eligible = [i for i in resting if abs(xyz[i, 2] - self.half_size) <= .008
                        and not bool(env.agent.is_grasping(env.all_cubes[i]).any())
                        and currently_visible(i)]
            while len(self.active) < requested_clusters:
                busy = {i for plan in self.active for i in plan['indices']}
                available = [i for i in eligible if i not in busy]
                if len(available) < 2:
                    break
                plan = plan_adjustment(
                    positions, available, resting, button, tcp, self.radius, self.rng,
                    reserved=self.active, slots=requested_clusters-len(self.active),
                    cube_waits=cube_waits, max_cluster_size=self.cluster_size[1])
                if plan is None:
                    break
                event = dict(start_step=step, duration_steps=DURATION, kind=plan["kind"],
                             cube_indices=plan["indices"], start_xy=plan["path"][0].tolist(),
                             planned_end_xy=plan["path"][-1].tolist(),
                             planned_yaw_delta=plan["yaw"].tolist(), outcome="moving",
                             batch_id=self.batch_id, requested_clusters=requested_clusters,
                             selection_anchor=plan['anchor'], selection_region=list(plan['region']),
                             selection_weights=[1 + cube_waits[i] for i in plan['indices']])
                self.events.append(event)
                plan.update(progress=0, event=event,
                            z=xyz[plan["indices"], 2].copy(),
                            q=[_array(env.all_cubes[i].pose.q).reshape(-1, 4)[0].copy() for i in plan["indices"]])
                self.active.append(plan)

        for active in list(self.active):
            if any(bool(env.agent.is_grasping(env.all_cubes[i]).any())
                   or abs(xyz[i, 2] - active["z"][j]) > .01
                   for j, i in enumerate(active['indices'])):
                self._finish(active, step, "stopped_for_contact")
                continue
            for j, i in enumerate(active['indices']):
                following = [*active['path'][active['progress']+1, j], active['z'][j]]
                q = planned_pose(active, j, (active['progress']+1)/DURATION)
                measurement = (self.visibility.measure(following, q) if currently_visible(i)
                               else visible[i])
                if not measurement['allowed']:
                    active['event']['stop_visibility'] = dict(cube_index=i, **measurement)
                    self._finish(active, step, "stopped_for_arm_occlusion")
                    break

        # Check all proposed movements before changing any actor. Recheck after a
        # group stops: its cubes become stationary obstacles to the other groups.
        while self.active:
            busy = {i for plan in self.active for i in plan['indices']}
            stationary = positions[[i for i in resting if i not in busy]]
            segments = [np.stack((positions[p['indices']], p['path'][p['progress']+1]))
                        for p in self.active]
            blocked = []
            for j, active in enumerate(self.active):
                others = [s for k, s in enumerate(segments) if k != j]
                moving = np.concatenate(others, axis=1) if others else None
                if not clear_path(segments[j], stationary, button, tcp, self.radius,
                                  gripper_clearance=STOP_CLEARANCE, moving_obstacles=moving):
                    blocked.append(active)
            if not blocked:
                break
            for active in blocked:
                self._finish(active, step, "stopped_for_clearance")

        import sapien

        for active in list(self.active):
            if active['progress'] == 0:
                # Reset waiting priority only when movement actually starts;
                # rejected plans keep their accumulated priority.
                for i in active['indices']:
                    self.cube_uses[i] = self.cube_uses.get(i, 0) + 1
                    self.last_moved_batch[i] = active['event']['batch_id']
            following = active['path'][active['progress'] + 1]
            for j, i in enumerate(active['indices']):
                actor = env.all_cubes[i]
                progress = (active["progress"] + 1) / DURATION
                q = planned_pose(active, j, progress)
                actor.set_pose(sapien.Pose([*following[j], active["z"][j]], q))
                actor.set_linear_velocity(np.zeros(3))
                actor.set_angular_velocity(np.zeros(3))
            active["progress"] += 1
            active["event"]["movement_steps"] = active["progress"]
            if active["progress"] == DURATION:
                self._finish(active, step + 1, "completed")
