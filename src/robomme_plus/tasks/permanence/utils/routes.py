"""Central circulation, robot-side button passage, and vacancy following."""

import heapq

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from robomme_plus.tasks.permanence.utils.moving import swept_gap

# The additional board-side space is filtered by board-sweep clearance and IK.
# Bounds describe complete footprints; planning subtracts the cover radius.
BOUNDS = np.array([[-0.49, 0.22], [-0.34, 0.34]])
ROBOT_BASE_XY = np.array([-0.615, 0.0])
ROBOT_BASE_RADIUS_M = 0.10


def point_segment_distance(points, starts, ends):
    points = np.array(points, dtype=float, copy=True)
    starts, ends = np.asarray(starts), np.asarray(ends)
    delta = ends - starts
    offset = points[..., None, :] - starts
    t = np.clip(np.sum(offset * delta, axis=-1) / np.maximum(np.sum(delta * delta, axis=-1), 1e-20), 0, 1)
    return np.linalg.norm(offset - t[..., None] * delta, axis=-1)


def segment_distance(before, after, starts, ends):
    """Distances between a moving center's entire segment and static segments."""
    before, after = np.asarray(before), np.asarray(after)
    starts, ends = np.asarray(starts), np.asarray(ends)
    distances = np.minimum(
        point_segment_distance(before, starts, ends),
        point_segment_distance(after, starts, ends),
    )
    if np.linalg.norm(after - before) < 1e-12:
        return distances
    distances = np.minimum(
        distances, point_segment_distance(starts, before[None], after[None])[:, 0]
    )
    distances = np.minimum(
        distances, point_segment_distance(ends, before[None], after[None])[:, 0]
    )
    direction, edges = after - before, ends - starts
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    denominator = cross(direction, edges)
    nonparallel = np.abs(denominator) > 1e-12
    safe = np.where(nonparallel, denominator, 1)
    t, u = (
        cross(starts - before, edges) / safe,
        cross(starts - before, direction) / safe,
    )
    intersects = nonparallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return np.where(intersects, 0, distances)


def point_box_distance(points, centers, half_sizes):
    offset = np.abs(np.asarray(points)[..., None, :]-centers)-half_sizes
    return np.linalg.norm(np.maximum(offset, 0), axis=-1)


def segment_box_distance(a, b, centers, half_sizes):
    corners = centers[:, None]+half_sizes[:, None]*np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    distance = segment_distance(a, b, corners.reshape(-1, 2),
                                np.roll(corners, -1, axis=1).reshape(-1, 2)).reshape(-1, 4).min(axis=1)
    return np.minimum(distance, np.minimum(point_box_distance(a, centers, half_sizes),
                                          point_box_distance(b, centers, half_sizes)))


CENTRAL_BOUNDS = np.array([[-0.36, 0.22], [-0.30, 0.30]])

MARGIN_M = 0.012


def initial_bounds(obstacles, board_front_x=.185, central_bounds=None):
    """Footprint bounds strictly between the buttons and the board plane."""
    bounds = np.array(CENTRAL_BOUNDS if central_bounds is None else central_bounds, dtype=float, copy=True)
    bounds[0, 0] = max(float(p[0]) + float(r) for p, r in obstacles) + MARGIN_M
    bounds[0, 1] = min(bounds[0, 1], board_front_x - MARGIN_M)
    return bounds


def passage_roles(count, passage_count, seed):
    """Uniform identities; a seeded common direction prevents head-on traffic."""
    if not 0 <= passage_count <= count:
        raise ValueError("Passage count must be between zero and the container count")
    role_rng = np.random.default_rng(seed + 1734005)
    indices = sorted(role_rng.choice(count, size=passage_count, replace=False).tolist())
    direction_rng = np.random.default_rng(seed + 1734006)
    direction = int(direction_rng.choice([-1, 1]))
    return {i: direction for i in indices}


def barriers(obstacles):
    buttons = np.unique(np.asarray([p for p, _ in obstacles], dtype=float), axis=0)
    if len(buttons) != 2:
        raise ValueError("Two distinct buttons are required")
    buttons = buttons[np.argsort(buttons[:, 1])]
    # Only connect the two buttons. There is no branch across the table.
    return buttons[:1], buttons[1:], np.array([0.055]), buttons


def consume_vacancy(vacancies, vacancy):
    """Remove this exact record without comparing dictionaries containing arrays."""
    index = next(i for i, entry in enumerate(vacancies) if entry is vacancy)
    return vacancies.pop(index)


def passage_intersects(before, after, x_max, y_range):
    """Whether any part of a segment enters the complete rear passage."""
    before, after = np.asarray(before), np.asarray(after)
    delta = after - before
    horizontal = abs(delta[..., 1]) < 1e-12
    denominator = np.where(horizontal, 1., delta[..., 1])
    a = (y_range[0] - before[..., 1]) / denominator
    b = (y_range[1] - before[..., 1]) / denominator
    start = np.where(horizontal, 0., np.maximum(0., np.minimum(a, b)))
    stop = np.where(horizontal, 1., np.minimum(1., np.maximum(a, b)))
    inside_y = (~horizontal | ((before[..., 1] >= y_range[0])
                              & (before[..., 1] <= y_range[1])))
    nearest_x = np.minimum(before[..., 0] + start * delta[..., 0],
                           before[..., 0] + stop * delta[..., 0])
    return inside_y & (start <= stop) & (nearest_x <= x_max)


def passage_forward(before, after, direction, x_max, y_range):
    """Allow waiting and sideways motion, but no reverse passage movement."""
    reverse = (np.asarray(after)[..., 1] - np.asarray(before)[..., 1]) * direction < -1e-12
    return ~(reverse & passage_intersects(before, after, x_max, y_range))


def rearrangement_audit(history):
    """Measure identities and relative ordering, rather than distance travelled."""
    path = np.asarray(history)
    first, last = path[0], path[-1]
    count = len(first)
    def neighbors(points):
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        return np.argsort(distances, axis=1)[:, :2]
    before, after = neighbors(first), neighbors(last)
    changed = np.any(np.sort(before, axis=1) != np.sort(after, axis=1), axis=1)
    initial_delta = first[:, None] - first[None, :]
    final_delta = last[:, None] - last[None, :]
    # Ignore nearly tied coordinates; small jitter does not count as overtaking.
    flips = ((initial_delta * final_delta < 0)
             & (abs(initial_delta) >= .025) & (abs(final_delta) >= .025))
    flip_counts = np.any(flips, axis=2).sum(axis=1)
    spans = np.linalg.norm(np.ptp(path, axis=0), axis=1)
    tail_spans = np.linalg.norm(np.ptp(path[-61:], axis=0), axis=1)
    return dict(
        initial_nearest_neighbors=before.tolist(),
        final_nearest_neighbors=after.tolist(),
        nearest_neighbor_changed=(before[:, 0] != after[:, 0]).tolist(),
        two_neighbor_set_changed=changed.tolist(),
        relative_order_changed_with_count=flip_counts.tolist(),
        endpoint_displacements_m=np.linalg.norm(last-first, axis=1).tolist(),
        actual_spatial_spans_m=spans.tolist(),
        last_60_step_spatial_spans_m=tail_spans.tolist(),
    )


def completed_passages(history, metadata):
    """Require arrival on both sides of the rear passage, not midline jitter."""
    path = np.asarray(history)
    if 'passage_assignments' in metadata:
        completed = []
        roles = {r['actor_index']: r for r in metadata['passage_routes']}
        events = metadata['passage_assignments']
        for k, event in enumerate(events):
            if event['event'] != 'assigned':
                continue
            i, start = event['actor_index'], event['step']
            stop = next((e['step'] for e in events[k+1:]
                         if e['actor_index'] == i and e['event'] == 'released'), len(path)-1)
            submeta = {key: value for key, value in metadata.items() if key != 'passage_assignments'}
            submeta['passage_routes'] = [roles[i]]
            for crossing in completed_passages(path[start:stop+1], submeta):
                crossing['entry_step'] += start
                crossing['exit_step'] += start
                completed.append(crossing)
        return sorted(completed, key=lambda event: event['exit_step'])
    rear = np.asarray(metadata['robot_side_crossing_xy_m'])
    completed = []
    for role in metadata['passage_routes']:
        i = role['actor_index']
        entry, exit = rear if role['direction'] == 'right-to-left' else rear[::-1]
        if 'one_way_passage' in metadata:
            # The passage is a lane, not an exact centerline. A safe lateral
            # detour still completes it when both end gates are visited.
            lane = metadata['one_way_passage']
            inside = path[:, i, 0] <= lane['x_max_m']
            entries = np.flatnonzero(inside & (abs(path[:, i, 1]-entry[1]) < .035))
            arrivals = np.flatnonzero(inside & (abs(path[:, i, 1]-exit[1]) < .035))
        else:
            entries = np.flatnonzero(np.linalg.norm(path[:, i]-entry, axis=1) < .035)
            arrivals = np.flatnonzero(np.linalg.norm(path[:, i]-exit, axis=1) < .035)
        if not len(entries):
            continue
        arrivals = arrivals[arrivals > entries[0]]
        # The route must actually use the rear crossing between those visits.
        for arrival in arrivals:
            section = path[entries[0]:arrival+1, i]
            middle = rear[:, 1].mean()
            crossings = np.flatnonzero((section[:-1, 1]-middle)*(section[1:, 1]-middle) < 0)
            if any(max(section[k, 0], section[k+1, 0]) < rear[0, 0]+.015 for k in crossings):
                completed.append(dict(actor_index=i, entry_step=int(entries[0]),
                                      exit_step=int(arrival), direction=role['direction']))
                break
    return completed


def motion_audit(routes):
    """Describe the executed script trace, separately from the longer plan."""
    history = np.asarray(routes.history)
    lengths = np.linalg.norm(np.diff(history, axis=0), axis=2).sum(axis=0)
    audit = dict(
        recorded_movement_steps=len(history) - 1,
        route_cursors=np.asarray(routes.cursors).tolist(),
        actual_travel_lengths_m=lengths.tolist(),
        actual_xy_spans_m=np.ptp(history, axis=0).tolist(),
        planned_travel_lengths_m=routes.metadata["route_lengths_m"],
        minimum_actual_continuous_pair_gap_m=routes.metadata.get(
            "minimum_actual_continuous_pair_gap_m"
        ),
        trace_file="motion_trace.npz",
    )
    if "route_exclusions" not in routes.metadata:
        return audit
    exclusions = routes.metadata["route_exclusions"]
    starts, ends, widths = [
        np.asarray(exclusions[k]) for k in ["starts_xy_m", "ends_xy_m", "radii_m"]
    ]
    robot_side_x = min(starts[0, 0], ends[0, 0]) - widths[0] - routes.radius - MARGIN_M
    middle_y = (starts[0, 1] + ends[0, 1]) / 2
    crossings, minimum_gap = [], float("inf")
    for step, (before, after) in enumerate(zip(history[:-1], history[1:]), start=1):
        for i, (a, b) in enumerate(zip(before, after)):
            minimum_gap = min(
                minimum_gap,
                float(
                    (
                        segment_distance(a, b, starts, ends) - widths - routes.radius
                    ).min()
                ),
            )
            if (a[1] - middle_y) * (b[1] - middle_y) < 0:
                fraction = (middle_y - a[1]) / (b[1] - a[1])
                x = a[0] + fraction * (b[0] - a[0])
                if x < robot_side_x:
                    crossings.append(
                        dict(
                            actor_index=i,
                            movement_step=step,
                            xy_m=[float(x), float(middle_y)],
                        )
                    )
    # Report only planned flow assignments that execution actually reached.
    issued = [
        event
        for event in routes.metadata["vacancy_flow_events"]
        if event["assigned_route_step"] < int(routes.cursors[event["mover_index"]])
    ]
    completed = [
        event
        for event in issued
        if any(
            np.linalg.norm(p[event["mover_index"]] - event["goal_xy_m"]) < 0.035
            for p in history[event["assigned_route_step"] :]
        )
    ]
    audit.update(
        robot_side_crossings=crossings,
        robot_side_crossing_actor_indices=sorted({c["actor_index"] for c in crossings}),
        minimum_actual_button_gap_clearance_m=minimum_gap,
        button_gap_avoided=minimum_gap >= MARGIN_M - 1e-4,
        reached_vacancy_flow_assignments=len(issued),
        completed_vacancy_flow_assignments=len(completed),
        full_plan_vacancy_flow_assignments=len(routes.metadata["vacancy_flow_events"]),
    )
    audit.update(rearrangement_audit(history))
    audit['completed_button_passages'] = completed_passages(history, routes.metadata)
    audit.update(counterflow_audit(history, routes.metadata))
    if 'one_way_passage' in routes.metadata:
        lane = routes.metadata['one_way_passage']
        allowed = passage_forward(history[:-1], history[1:], lane['direction'],
                                  lane['x_max_m'], lane['y_range_m'])
        audit['reverse_passage_moves'] = int(np.count_nonzero(~allowed))
        intrusions = passage_intersects(history[:-1], history[1:], lane['x_max_m'], lane['y_range_m'])
        audit['central_passage_intrusions'] = int(np.count_nonzero(
            intrusions & central_membership(len(history)-1, history.shape[1], routes.metadata)))
    return audit


def central_membership(steps, count, metadata):
    """Per-step traffic roles, including handovers during continuous motion."""
    central = np.zeros((steps, count), dtype=bool)
    central[:, metadata['central_actor_indices']] = True
    for event in metadata.get('passage_assignments', []):
        central[event['step']:, event['actor_index']] = event['event'] == 'released'
    return central


def receiving_side_velocities(local, receiving, direction, limits):
    """Add one shared B->A drift, keeping local variation and speed limits.

    The mean preferred velocity of the receiving-side group points toward A.
    Individual members may still reverse. Collision constraints are applied
    afterwards and can override a preferred move.
    """
    result = np.array(local, dtype=float, copy=True)
    receiving = np.asarray(receiving, dtype=bool)
    if not receiving.any():
        return result
    group, caps = result[receiving], np.asarray(limits)[receiving]

    def shifted(amount):
        velocity = group - np.array([0., direction * amount])
        norm = np.linalg.norm(velocity, axis=1)
        return velocity * np.minimum(1., caps / np.maximum(norm, 1e-12))[:, None]

    # Every member gets the common drift; increase it only if the group's
    # local steering would otherwise overwhelm the B->A trend.
    low, high = .35 * caps.mean(), 3 * caps.max()
    target = -.25 * caps.mean()
    if np.mean(shifted(low)[:, 1] * direction) > target:
        for _ in range(24):
            middle = (low + high) / 2
            if np.mean(shifted(middle)[:, 1] * direction) > target:
                low = middle
            else:
                high = middle
        low = high
    result[receiving] = shifted(low)
    return result


def counterflow_audit(history, metadata):
    """Measure actual B->A velocity using membership before each move."""
    path = np.asarray(history)
    indices = metadata['central_actor_indices']
    direction = metadata['traffic_direction']
    middle = np.asarray(metadata['robot_side_crossing_xy_m'])[:, 1].mean()
    receiving = ((path[:-1, indices, 1] - middle) * direction > 0
                 ) & central_membership(len(path)-1, path.shape[1], metadata)[:, indices]
    toward_a = -direction * np.diff(path[:, indices, 1], axis=0)
    net = np.sum(np.where(receiving, toward_a, 0), axis=1)
    crossed = [int(i) for i in indices if np.any(
        (path[:, i, 1] - middle) * direction > .04) and any(
            (path[k, i, 1]-middle)*direction > .04
            and np.any((path[k+1:, i, 1]-middle)*direction < -.04)
            for k in range(len(path)-1))]
    return dict(
        receiving_side_net_velocity_toward_a_m_per_step=net.tolist(),
        receiving_side_total_motion_toward_a_m=float(net.sum()),
        receiving_side_reverse_local_moves=int(np.sum(receiving & (toward_a < -1e-6))),
        redistributed_actor_indices=crossed,
    )


def plan_routes(
    count,
    radius,
    seed,
    steps=720,
    speed=0.0045,
    obstacles=(),
    initial=None,
    initial_reject=None,
    reachable_area=None,
    geometry_only=False,
    passage_count=2,
    extra_obstacles=(),
    initial_board_front_x=.185,
    continuous=False,
    workspace=None,
    pickup_bypass_margin=0.,
    pickup_passing=False,
    rectangle_obstacles=(),
):
    if not 0 <= pickup_bypass_margin <= .10 or (pickup_bypass_margin and not continuous):
        raise ValueError('Pickup bypass requires continuous motion and a margin between 0 and 10 cm')
    if pickup_passing and not continuous:
        raise ValueError('Pickup passing requires continuous motion')
    rng = np.random.default_rng(seed)
    starts, ends, widths, buttons = barriers(obstacles)
    button_obstacles = [(np.asarray(p), float(r)) for p, r in obstacles]
    extra_obstacles = [(np.asarray(p), float(r)) for p, r in extra_obstacles]
    bounds = np.array(BOUNDS if workspace is None else workspace['bounds_xy_m'], dtype=float)
    central_bounds = np.array(CENTRAL_BOUNDS if workspace is None else workspace['central_bounds_xy_m'], dtype=float)
    if (bounds.shape != (2, 2) or central_bounds.shape != (2, 2)
            or not np.isfinite(bounds).all() or not np.isfinite(central_bounds).all()
            or np.any(bounds[:, 1] <= bounds[:, 0])
            or np.any(central_bounds[:, 1] <= central_bounds[:, 0])
            or np.any(central_bounds[:, 0] < bounds[:, 0])
            or np.any(central_bounds[:, 1] > bounds[:, 1])):
        raise ValueError('Central motion bounds must fit inside the outer workspace')
    start_bounds = initial_bounds(button_obstacles, initial_board_front_x, central_bounds)
    obstacles = [*button_obstacles, *extra_obstacles,
                 (ROBOT_BASE_XY, ROBOT_BASE_RADIUS_M)]
    centers = np.array([p for p, _ in obstacles])
    sizes = np.array([r for _, r in obstacles])
    box_centers = np.asarray([p for p, _ in rectangle_obstacles], dtype=float).reshape(-1, 2)
    box_half_sizes = np.asarray([h for _, h in rectangle_obstacles], dtype=float).reshape(-1, 2)
    low, high = bounds[:, 0] + radius, bounds[:, 1] - radius
    center_low, center_high = (
        central_bounds[:, 0] + radius,
        central_bounds[:, 1] - radius,
    )
    allowance = radius + MARGIN_M + speed
    # Reserve the entire rear lane, including a buffer around each mouth.
    # Central counterflow must not enter an exit-side pocket and oppose it.
    passage_x_max = float(buttons[:, 0].min() - .055 - allowance + .015)
    passage_y_range = np.array([buttons[0, 1] - .055 - allowance - .045,
                                buttons[1, 1] + .055 + allowance + .045])
    # The deeper central rectangle reaches behind some sampled buttons.
    # Close only its rear crossing to non-travelers, retaining usable space
    # on either side without granting them an unassigned around-button route.
    middle_y = float(buttons[:, 1].mean())
    central_barrier_start = np.array([[bounds[0, 0], middle_y]])
    central_barrier_end = np.array([[buttons[:, 0].min() - widths[0], middle_y]])
    rear_nodes = None
    lane_clearance = 2*radius + MARGIN_M + .012
    bypass_indices, returning_indices = set(), set()
    bypass_events, outside_last = [], set()
    passing_goals, passing_events = {}, []
    actor_radii = np.full(count, radius)
    parked_obstacles = []

    def pair_clearance(index, extra=.001):
        if not pickup_passing:
            return 2*radius + MARGIN_M + extra
        return actor_radii[index] + actor_radii[np.arange(count) != index] + MARGIN_M + extra

    def valid(points, central=False, *, index=None, expanded=False):
        lower, upper = (center_low, center_high) if central else (low, high)
        if expanded or index in bypass_indices:
            lower, upper = lower - pickup_bypass_margin, upper + pickup_bypass_margin
        result = (
            reachable_area.contains(points)
            & np.all((points >= lower) & (points <= upper), axis=-1)
            & np.all(
                point_segment_distance(points, starts, ends) > widths + allowance,
                axis=-1,
            )
            & np.all(
                np.linalg.norm(points[..., None, :] - centers, axis=-1)
                > sizes + allowance,
                axis=-1,
            )
            & np.all(point_box_distance(points, box_centers, box_half_sizes) > allowance, axis=-1)
        )
        if central:
            result &= np.all(point_segment_distance(points, central_barrier_start,
                                                     central_barrier_end) > allowance, axis=-1)
            result &= ~passage_intersects(points, points, passage_x_max, passage_y_range)
            if continuous and rear_nodes is not None:
                result &= point_segment_distance(points, rear_nodes[:1], rear_nodes[1:])[..., 0] > lane_clearance
        return result

    def visible(a, b, central=False, *, index=None):
        if np.linalg.norm(a - b) < 1e-12:
            return bool(valid(a, central=central, index=index))
        return (
            reachable_area.segment(a, b)
            and (not central or np.all(segment_distance(
                a, b, central_barrier_start, central_barrier_end) > allowance))
            and (not central or not passage_intersects(a, b, passage_x_max, passage_y_range))
            and (not continuous or not central or rear_nodes is None
                 or np.all(segment_distance(a, b, rear_nodes[:1], rear_nodes[1:]) > lane_clearance))
            and np.all(segment_distance(a, b, starts, ends) > widths + allowance)
            and np.all(
                point_segment_distance(centers, a[None], b[None])[:, 0]
                > sizes + allowance
            )
            and np.all(segment_box_distance(a, b, box_centers, box_half_sizes) > allowance)
        )

    # 1 cm of extra room beyond the moving container's allowance.
    outer_y = np.array(
        [
            buttons[0, 1] - 0.055 - allowance - 0.010,
            buttons[1, 1] + 0.055 + allowance + 0.010,
        ]
    )
    edge_y = [center_low[1] + 0.006, center_high[1] - 0.006]
    front_nodes = np.array(
        [
            [-0.03, edge_y[0]],
            [-0.03, edge_y[1]],
            [-0.03, 0],
            [max(center_low[0] + .006, passage_x_max + .006), edge_y[0]],
            [max(center_low[0] + .006, passage_x_max + .006), edge_y[1]],
        ]
    )
    # A fixed x=-.405 corridor has an unreachable middle section for this Panda
    # pose. Select a robot-side crossing whose entire segment is reachable;
    # otherwise graph search silently substitutes a long far-side detour.
    nearest_rear_x = buttons[:, 0].min() - 0.055 - allowance - 0.006
    rear_nodes = None
    for rear_x in np.linspace(nearest_rear_x, low[0], 100):
        candidate = np.array([[rear_x, outer_y[0]], [rear_x, outer_y[1]]])
        if np.all(valid(candidate)) and visible(*candidate):
            rear_nodes = candidate
            break
    if rear_nodes is None:
        raise ValueError("No reachable robot-side crossing around the buttons")
    if continuous:
        front_nodes[3:, 0] = max(front_nodes[3, 0], rear_nodes[0, 0]+lane_clearance+.008)
    if not np.all(valid(front_nodes, central=True)) or not np.all(valid(rear_nodes)):
        raise ValueError("Button placement leaves no safe route")
    # Individually valid corners can still be isolated by a button, especially
    # with the narrower width and hard's larger movement allowance. Reject
    # that button proposal before packing containers into disconnected lanes.
    connected = {0}
    while True:
        neighbors = {j for j in range(len(front_nodes)) if j not in connected
                     and any(visible(front_nodes[i], front_nodes[j], central=True)
                             for i in connected)}
        if not neighbors:
            break
        connected.update(neighbors)
    if len(connected) != len(front_nodes):
        raise ValueError("Button placement disconnects central movement lanes")
    if geometry_only:
        if count:
            # Button placement must leave room for every initial container
            # between the buttons and board, without widening that region.
            packed_initial(count, radius, seed, speed, button_obstacles,
                           reachable_area=reachable_area,
                           extra_obstacles=extra_obstacles, bounds=start_bounds,
                           extra_spacing=.001 if workspace is not None else None)
        return rear_nodes, front_nodes
    directions = passage_roles(count, passage_count, seed)
    passage_indices = list(directions)
    passage_set = set(passage_indices)
    central_indices = [i for i in range(count) if i not in passage_set]
    visitor_loop = np.concatenate((rear_nodes, front_nodes[[1, 2, 0]]))
    # Positive Y is the left button's side; each loop starts at the entry to
    # its chosen rear crossing, then exits through the matching central lane.
    visitor_loops = {
        i: visitor_loop if direction == 1 else visitor_loop[[1, 0, 4, 3, 2]]
        for i, direction in directions.items()
    }

    passage_direction = next(iter(directions.values()), 1)

    def forward_crossing(a, b):
        return passage_forward(a, b, passage_direction, passage_x_max, passage_y_range)

    def navigation(a, goal, index):
        nodes = np.concatenate((rear_nodes, front_nodes)) if index in passage_set else front_nodes
        vertices = np.concatenate(([a], nodes, [goal]))
        distance, queue, parents = {0: 0.0}, [(0.0, 0)], {}
        while queue:
            cost, i = heapq.heappop(queue)
            if cost != distance[i]:
                continue
            if i == len(vertices) - 1:
                route = [vertices[i]]
                while parents[i] != 0:
                    i = parents[i]
                    route.append(vertices[i])
                return route[::-1]
            for j in range(1, len(vertices)):
                candidate = cost + np.linalg.norm(vertices[i] - vertices[j])
                if (
                    j != i
                    and candidate < distance.get(j, float("inf"))
                    and visible(vertices[i], vertices[j], central=index not in passage_set, index=index)
                    and forward_crossing(vertices[i], vertices[j])
                ):
                    distance[j], parents[j] = candidate, i
                    heapq.heappush(queue, (candidate, j))
        # The dense traffic graph can find detours missed by the legacy
        # handful of static waypoints.
        return [np.asarray(goal)]

    def sample(index):
        for _ in range(100):
            pool = rng.uniform(center_low, center_high, (96, 2))
            pool = pool[valid(pool, central=True)]
            others = points[np.arange(count) != index]
            if not len(pool):
                continue
            distances = np.linalg.norm(pool[:, None] - others, axis=2)
            reservations = [t[-1] for i, t in enumerate(targets) if i != index and t]
            free = np.linalg.norm(pool - points[index], axis=1) > .12
            pool, distances = pool[free], distances[free]
            if not len(pool):
                continue
            density = np.exp(-(distances / .16)**2).sum(axis=1)
            if reservations:
                density += .5 * np.exp(-(np.linalg.norm(pool[:, None] - reservations,
                                                       axis=2) / .12)**2).sum(axis=1)
            for k in np.argsort(density + rng.uniform(0, 1.2, len(pool))):
                p = pool[k]
                if any(visible(p, node, central=True) for node in front_nodes):
                    return p
        raise ValueError("No free central destination")

    # Check each complete simultaneous step below, rather than reserve two
    # extra speed increments that close otherwise traversable lanes.
    separation = 2 * radius + MARGIN_M + .001
    if initial is None:
        initial_geometry = {}
        def reject_start(index, point):
            key = tuple(point)
            if key not in initial_geometry:
                initial_geometry[key] = (valid(point, central=True)
                    and any(visible(point, node, central=True) for node in front_nodes))
            return (not initial_geometry[key]
                    or (initial_reject is not None and initial_reject(index, point)))

        initial, packing_attempts = packed_initial(
            count, radius, seed, speed, button_obstacles, reject_start, reachable_area,
            extra_obstacles=extra_obstacles, bounds=start_bounds,
            extra_spacing=.001 if workspace is not None else None,
        )
    else:
        packing_attempts = 0
    points = initial
    points = np.array(points, dtype=float, copy=True)
    if points.shape != (count, 2) or not np.all(valid(points, central=True)):
        raise ValueError("Initial footprint must be central and outside the button gap")
    if not np.all((points >= start_bounds[:, 0] + radius)
                  & (points <= start_bounds[:, 1] - radius)):
        raise ValueError("Initial footprints must lie between the buttons and board")
    if swept_gap(points, points, np.full(count, radius)) < MARGIN_M:
        raise ValueError("Initial container footprints overlap")
    if passage_count:
        # Route roles are assigned after packing, before RGB/goal identities.
        # Starting on the opposite side can consume the entire movement
        # window before even reaching the entrance. Use the nearest safe
        # approaches to the randomly chosen entry side.
        entry = rear_nodes[0 if passage_direction == 1 else 1]
        approach_costs = []
        for point in points:
            approach = navigation(point, entry, passage_indices[0])
            approach_costs.append(np.linalg.norm(np.diff(np.array([point, *approach]), axis=0), axis=1).sum())
        passage_indices = sorted(np.argsort(approach_costs, kind='stable')[:passage_count].tolist())
        passage_set = set(passage_indices)
        central_indices = [i for i in range(count) if i not in passage_set]
        directions = dict.fromkeys(passage_indices, passage_direction)
        visitor_loops = {i: visitor_loop if passage_direction == 1 else visitor_loop[[1, 0, 4, 3, 2]]
                         for i in passage_indices}

    paths, headings, targets = (
        [points.copy()],
        np.zeros((count, 2)),
        [[] for _ in points],
    )
    loop_indices = {i: 0 for i in passage_indices}
    angles = np.linspace(0, 2 * np.pi, 41, endpoint=False)
    units = np.column_stack((np.cos(angles), np.sin(angles)))
    minimum = float("inf")
    visitor_inside = dict.fromkeys(passage_indices, True)
    last_central = points.copy()
    vacancies, flow_events, filling = [], [], {}
    velocity_flow = []
    receiving_flow = np.zeros(count, dtype=bool)
    yielding = {}
    departed = set()
    escapes, recovery_events = {}, []
    recovery_steps = np.full(count, -30)
    velocities = np.zeros_like(points)
    passage_assignments = [dict(actor_index=i, step=0, event='assigned') for i in passage_indices]
    passage_counts = np.zeros(count, dtype=int)
    passage_counts[passage_indices] = 1
    last_passage = np.full(count, -1000)
    active_passages = passage_indices.copy()

    def rotate_visitors(step, frozen):
        """Hand the lane to new covers as soon as a visitor clears its exit."""
        for i in list(passage_indices):
            # Rejoin central traffic only after the complete rear passage and
            # its mouth are behind the cover. A grasped cover outside the lane
            # also gives up its slot so manipulation cannot block recruitment.
            exited = loop_indices[i] >= 3 and valid(points[i], central=True)
            if not exited and not (i in frozen and valid(points[i], central=True)):
                continue
            passage_indices.remove(i)
            passage_set.remove(i)
            central_indices.append(i)
            last_passage[i] = step
            passage_assignments.append(dict(actor_index=i, step=step, event='released'))
            targets[i] = []
            fields.pop(i, None)
            escapes.pop(i, None)
            receiving_flow[i] = True
        entry = rear_nodes[0 if passage_direction == 1 else 1]
        while len(passage_indices) < passage_count:
            eligible = [i for i in central_indices if i not in frozen and step-last_passage[i] >= 60]
            if not eligible:
                break
            # Prefer available space near the entrance, with age/fairness and
            # seeded variation so a permanent pair cannot monopolize the lane.
            costs = [np.linalg.norm(points[i]-entry) + .18*passage_counts[i]
                     + rng.uniform(0, .055) for i in eligible]
            i = eligible[int(np.argmin(costs))]
            central_indices.remove(i)
            passage_indices.append(i)
            passage_set.add(i)
            directions[i] = passage_direction
            visitor_loops[i] = visitor_loop if passage_direction == 1 else visitor_loop[[1, 0, 4, 3, 2]]
            loop_indices[i] = 0
            visitor_inside[i] = True
            last_central[i] = points[i].copy()
            departed.discard(i)
            targets[i] = []
            fields.pop(i, None)
            filling.pop(i, None)
            yielding.pop(i, None)
            escapes.pop(i, None)
            receiving_flow[i] = False
            passage_counts[i] += 1
            passage_assignments.append(dict(actor_index=i, step=step, event='assigned'))
        active_passages[:] = passage_indices

    # A local shortest-path field accounts for other covers, instead of
    # repeatedly steering straight into them along the fixed waypoint graph.
    grid_spacing = .012
    grid_x = np.arange(low[0], high[0] + 1e-9, grid_spacing)
    grid_y = np.arange(low[1], high[1] + 1e-9, grid_spacing)
    if pickup_bypass_margin:
        extra = grid_spacing * np.arange(1, int(np.ceil(pickup_bypass_margin/grid_spacing)) + 1)
        grid_x = np.r_[grid_x[0]-extra[::-1], grid_x, grid_x[-1]+extra]
        grid_y = np.r_[grid_y[0]-extra[::-1], grid_y, grid_y[-1]+extra]
    grid = np.stack(np.meshgrid(grid_x, grid_y, indexing='ij'), axis=-1).reshape(-1, 2)
    nx, ny = len(grid_x), len(grid_y)
    ids = np.arange(len(grid)).reshape(nx, ny)
    sources, destinations = [], []
    for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a = ids[:nx-dx, max(0, -dy):ny-max(0, dy)].ravel()
        b = ids[dx:, max(0, dy):ny-max(0, -dy)].ravel()
        sources.extend(a); destinations.extend(b)
    sources, destinations = np.array(sources), np.array(destinations)
    fine_start = fine_end = len(grid)
    if pickup_passing:
        # Keep the original graph for ordinary traffic. Add a half-spacing
        # graph only for actors detouring around a pickup, so a narrow but
        # usable gap is not lost between 12 mm grid vertices.
        fine_x = np.linspace(grid_x[0], grid_x[-1], 2*len(grid_x)-1)
        fine_y = np.linspace(grid_y[0], grid_y[-1], 2*len(grid_y)-1)
        fine = np.stack(np.meshgrid(fine_x, fine_y, indexing='ij'), axis=-1).reshape(-1, 2)
        fine_ids = np.arange(len(fine)).reshape(len(fine_x), len(fine_y)) + len(grid)
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            a = fine_ids[:len(fine_x)-dx, max(0, -dy):len(fine_y)-max(0, dy)].ravel()
            b = fine_ids[dx:, max(0, dy):len(fine_y)-max(0, -dy)].ravel()
            sources = np.r_[sources, a]
            destinations = np.r_[destinations, b]
        sources = np.r_[sources, ids.ravel()]
        destinations = np.r_[destinations, fine_ids[::2, ::2].ravel()]
        grid = np.r_[grid, fine]
        fine_end = len(grid)
    # IK can leave a passage narrower than the generic grid spacing. Keep
    # the exact fully validated corridor in the graph instead of deleting it
    # accidentally through grid rounding.
    corridor = np.linspace(rear_nodes[0], rear_nodes[1],
                           int(np.ceil(np.linalg.norm(rear_nodes[1]-rear_nodes[0])/.008))+1)
    offset = len(grid)
    distance = np.linalg.norm(corridor[:, None]-grid, axis=2)
    a, b = np.where(distance < .03)
    sources = np.r_[sources, offset+a, np.arange(offset, offset+len(corridor)-1)]
    destinations = np.r_[destinations, b, np.arange(offset+1, offset+len(corridor))]
    grid = np.r_[grid, corridor]
    edge_lengths = np.linalg.norm(grid[sources] - grid[destinations], axis=1)
    regions = [(central, expanded) for central in (True, False)
               for expanded in ((False, True) if pickup_bypass_margin or pickup_passing else (False,))]
    static_masks = {(central, expanded): valid(grid, central=central, expanded=expanded)
                    for central, expanded in regions}
    static_edges = {}
    for central, expanded in regions:
        key = central, expanded
        mask = static_masks[key]
        if not expanded:
            mask[fine_start:fine_end] = False
        edges = mask[sources] & mask[destinations]
        for t in [.25, .5, .75]:
            edges &= valid((1-t)*grid[sources] + t*grid[destinations], central=central, expanded=expanded)
        static_edges[key] = edges
        if central:
            static_edges[key] &= ~passage_intersects(
                grid[sources], grid[destinations], passage_x_max, passage_y_range)
    fields = {}

    def crowding_cost(index, candidates):
        # A shared flow must not simply transfer a queue to the departure
        # side. All central covers prefer nearby empty space, on either side.
        # The gradient acts before contact, including when a move approaches A.
        others = points[np.arange(count) != index]
        distances = np.linalg.norm(candidates[:, None] - others, axis=2)
        weight = .006 if index in passage_set else .045
        return weight * np.exp(-(distances / .13)**2).sum(axis=1)

    def local_links(point, nodes, index):
        central = index not in passage_set
        delta = nodes - point
        okay = np.ones(len(nodes), dtype=bool)
        if central:
            okay &= ~passage_intersects(point, nodes, passage_x_max, passage_y_range)
        for t in [.25, .5, .75]:
            okay &= valid(point + t * delta, central=central, index=index)
        others = points[np.arange(count) != index]
        relative = others[None] - point
        t = np.clip(np.sum(relative * delta[:, None], axis=2)
                    / np.maximum(np.sum(delta * delta, axis=1)[:, None], 1e-20), 0, 1)
        okay &= np.all(np.linalg.norm(relative - t[..., None]*delta[:, None], axis=2) > pair_clearance(index), axis=1)
        if pickup_passing:
            for center, parked_radius in parked_obstacles:
                required = min(actor_radii[index]+parked_radius+MARGIN_M,
                               float(np.linalg.norm(point-center)))
                closest = point_segment_distance(np.asarray(center),
                                                 np.broadcast_to(point, nodes.shape), nodes)
                okay &= closest >= required-1e-10
        return okay

    def traffic_costs(index, candidates, step):
        goal = targets[index][-1]
        key = tuple(goal)
        cache = fields.get(index)
        if cache is None or step-cache[0] >= 6 or cache[1] != key:
            central = index not in passage_set
            region = central, index in bypass_indices
            others = points[np.arange(count) != index]
            free = static_masks[region] & np.all(np.linalg.norm(grid[:, None] - others, axis=2) > pair_clearance(index), axis=1)
            if pickup_passing:
                for center, parked_radius in parked_obstacles:
                    free &= np.linalg.norm(grid-center, axis=1) > actor_radii[index]+parked_radius+MARGIN_M
            edges = static_edges[region] & free[sources] & free[destinations]
            candidates_edges = np.flatnonzero(edges)
            midpoints = (grid[sources[candidates_edges]] + grid[destinations[candidates_edges]])/2
            edges[candidates_edges] &= np.all(np.linalg.norm(midpoints[:, None] - others, axis=2) > pair_clearance(index), axis=1)
            if pickup_passing:
                for center, parked_radius in parked_obstacles:
                    edges[candidates_edges] &= point_segment_distance(np.asarray(center),
                        grid[sources[candidates_edges]], grid[destinations[candidates_edges]]) > actor_radii[index]+parked_radius+MARGIN_M
            s, e, weights = sources[edges], destinations[edges], edge_lengths[edges]
            ss, ee, ww = np.r_[s,e], np.r_[e,s], np.r_[weights,weights]
            keep = forward_crossing(grid[ss], grid[ee])
            graph = csr_matrix((ww[keep], (ss[keep], ee[keep])), shape=(len(grid), len(grid)))
            choices = np.flatnonzero(free & (np.asarray(graph.getnnz(axis=1)) > 0))
            if not len(choices):
                return np.linalg.norm(candidates - targets[index][0], axis=1)
            local = choices[np.linalg.norm(grid[choices] - points[index], axis=1) < .06]
            local = local[local_links(points[index], grid[local], index)]
            if not len(local):
                return np.linalg.norm(candidates - targets[index][0], axis=1)
            start = local[np.argmin(np.linalg.norm(grid[local] - points[index], axis=1))]
            from_start = dijkstra(graph, directed=True, indices=int(start))
            component = choices[np.isfinite(from_start[choices])]
            end = component[np.argmin(np.linalg.norm(grid[component] - goal, axis=1))]
            distances = dijkstra(graph.T, directed=True, indices=int(end))
            cache = (step, key, distances, grid[end])
            fields[index] = cache
        distances = cache[2]
        nearby = np.flatnonzero(np.isfinite(distances) & (np.linalg.norm(grid - points[index], axis=1) < .05))
        nearby = nearby[local_links(points[index], grid[nearby], index)]
        if not len(nearby):
            return np.linalg.norm(candidates - targets[index][0], axis=1)
        # A grid approximation must not strand a cover at its final cell.
        # Use the exact goal once the complete remaining segment is free.
        if (np.linalg.norm(points[index]-goal) < .08
                and visible(points[index], goal, central=index not in passage_set, index=index)
                and local_links(points[index], np.array([goal]), index)[0]
                and forward_crossing(points[index], goal)):
            return np.linalg.norm(candidates-goal, axis=1)
        links = forward_crossing(candidates[:, None], grid[nearby])
        return np.min(np.where(links, np.linalg.norm(candidates[:, None] - grid[nearby], axis=2) + distances[nearby], np.inf), axis=1)

    def escape(index, step):
        """Move sideways/back into a genuinely free footprint before retrying."""
        destinations = np.concatenate([points[index]+d*units for d in (.06, .09, .13, .18)])
        others = points[np.arange(count) != index]
        nearest = np.linalg.norm(destinations[:, None]-others, axis=2).min(axis=1)
        allowed = valid(destinations, central=index not in passage_set, index=index)
        allowed &= (nearest > separation+.005) & forward_crossing(points[index], destinations)
        choices = np.flatnonzero(allowed)
        choices = [k for k in choices if local_links(points[index], destinations[k:k+1], index)[0]
                   and visible(points[index], destinations[k], central=index not in passage_set, index=index)]
        if not choices:
            return False
        score = nearest[choices] + .5*np.linalg.norm(destinations[choices]-points[index], axis=1)
        score += rng.uniform(0, .04, len(choices))
        k = choices[int(np.argmax(score))]
        if index not in escapes:
            escapes[index] = (step, targets[index])
        targets[index] = [destinations[k]]
        fields.pop(index, None)
        filling.pop(index, None)
        yielding.pop(index, None)
        recovery_events.append(dict(actor_index=int(index), step=int(step),
                                    origin_xy_m=points[index].tolist(), goal_xy_m=destinations[k].tolist()))
        return True

    def vacancy_goal(step):
        # Once a visitor leaves, move a nearby cover into the vacated footprint.
        # Its old footprint becomes the next vacancy, producing a short chain.
        for vacancy in list(vacancies):
            if (np.linalg.norm(points - vacancy['xy'], axis=1).min() < separation
                    or step - vacancy['created_step'] > 40):
                consume_vacancy(vacancies, vacancy)
                continue
            candidates = [
                i
                for i in central_indices
                if i not in filling
                and i not in yielding
                and i not in escapes
                and i != vacancy.get("previous_mover")
                and np.linalg.norm(points[i] - vacancy["xy"]) > 0.045
                and np.linalg.norm(points[i] - vacancy["xy"]) < 0.28
                and visible(points[i], vacancy["xy"], central=True)
            ]
            if not candidates:
                continue
            i = min(candidates, key=lambda j: np.linalg.norm(points[j] - vacancy["xy"]))
            event = dict(
                visitor_index=vacancy["visitor"],
                mover_index=int(i),
                goal_xy_m=vacancy["xy"].tolist(),
                origin_xy_m=points[i].tolist(),
                assigned_route_step=step,
                chain_depth=vacancy["depth"],
            )
            flow_events.append(event)
            filling[i] = event
            targets[i] = navigation(points[i], vacancy["xy"], i)
            consume_vacancy(vacancies, vacancy)

    def clear_entry(step):
        # Clear only the lane a visitor is about to use on returning from the
        # rear passage; neighbors nudge into nearby free central footprints.
        for visitor in passage_indices:
            if not targets[visitor]:
                continue
            entry_a, entry_b = points[visitor], targets[visitor][0]
            if np.linalg.norm(entry_b - entry_a) > .22:
                entry_b = entry_a + .22 * (entry_b - entry_a) / np.linalg.norm(entry_b - entry_a)
            for i in central_indices:
                if i in yielding or i in escapes or np.linalg.norm(points[i] - entry_a) > .24:
                    continue
                distance = point_segment_distance(
                    points[i], entry_a[None], entry_b[None]
                )[0]
                if distance > 2 * radius + MARGIN_M + 0.018:
                    continue
                destinations = np.concatenate(
                    [points[i] + d * units for d in [0.07, 0.10, 0.14, 0.18]]
                )
                valid_dest = valid(destinations, central=True, index=i)
                lane_distance = point_segment_distance(
                    destinations, entry_a[None], entry_b[None]
                )[:, 0]
                nearest = np.linalg.norm(
                    destinations[:, None] - points[np.arange(count) != i], axis=2
                ).min(axis=1)
                choices = np.flatnonzero(
                    valid_dest
                    & (lane_distance > 2 * radius + MARGIN_M + 0.025)
                    & (nearest > separation)
                )
                reservations = [t[-1] for j, t in enumerate(targets) if j != i and t]
                if reservations:
                    choices = [k for k in choices if np.linalg.norm(
                        destinations[k] - reservations, axis=1).min() > separation]
                choices = [k for k in choices if visible(points[i], destinations[k], central=True, index=i)]
                if not choices:
                    continue
                k = min(
                    choices,
                    key=lambda k: (
                        np.linalg.norm(destinations[k] - points[i]) - 0.2 * nearest[k]
                    ),
                )
                event = dict(
                    visitor_index=visitor,
                    mover_index=i,
                    goal_xy_m=destinations[k].tolist(),
                    origin_xy_m=points[i].tolist(),
                    assigned_route_step=step,
                    type="clear_return_lane",
                )
                filling.pop(i, None)
                yielding[i] = event
                flow_events.append(event)
                targets[i] = navigation(points[i], destinations[k], i)

    def stream():
        import itertools
        nonlocal minimum
        for step in itertools.count():
            feedback = yield points.copy()
            frozen = set()
            parked = ()
            blockers = set()
            speed_scale = 1.
            allow_contact_recovery = False
            if feedback is not None:
                actual, frozen = feedback[:2]
                if len(feedback) > 2:
                    parked = feedback[2]
                if len(feedback) > 3:
                    blockers = set(feedback[3].get('pickup_blockers', ()))
                    speed_scale = feedback[3].get('speed_scale', 1.)
                    allow_contact_recovery = feedback[3].get('allow_contact_recovery', False)
                    if not np.isfinite(speed_scale) or not 0 < speed_scale <= 1:
                        raise ValueError('Traffic speed scale must be in (0, 1]')
                    if pickup_passing:
                        radii = np.asarray(feedback[3].get('actor_radii_m', np.full(count, radius)))
                        if radii.shape != (count,) or not np.isfinite(radii).all() or np.any(radii <= 0) or np.any(radii > radius):
                            raise ValueError('Actor radii must be positive and no larger than the full container footprint')
                        if np.any(radii != actor_radii):
                            fields.clear()
                        actor_radii[:] = radii
                if pickup_passing:
                    parked_obstacles[:] = parked
                points[:] = actual
                frozen = set(frozen)
            before = points.copy()
            if continuous:
                rotate_visitors(step, frozen)
            if pickup_bypass_margin or pickup_passing:
                admitted = set()
                for i in range(count):
                    lower, upper = (center_low, center_high) if i not in passage_set else (low, high)
                    outside = np.any(points[i] < lower+.003) or np.any(points[i] > upper-.003)
                    detouring = i in bypass_indices and outside
                    near = (i not in frozen and any(
                        j != i and np.linalg.norm(points[i]-points[j]) < .22 for j in blockers))
                    if near or detouring:
                        admitted.add(i)
                    if detouring and not near and i not in returning_indices and i not in frozen:
                        # Goals stay inside the nominal area. Finish a detour
                        # smoothly after the pickup instead of clipping poses.
                        targets[i] = navigation(points[i], sample(i), i)
                        fields.pop(i, None)
                        escapes.pop(i, None)
                        filling.pop(i, None)
                        yielding.pop(i, None)
                        returning_indices.add(i)
                    if not outside:
                        returning_indices.discard(i)
                for i in bypass_indices ^ admitted:
                    fields.pop(i, None)
                bypass_indices.clear()
                bypass_indices.update(admitted)
            # Carry each crossing beyond the center line before releasing its
            # drift. Otherwise an old B-side destination pulls it back across the
            # switch on every other frame, creating an artificial queue at center.
            side = (before[:, 1] - middle_y) * passage_direction
            arrived = receiving_flow & (side < -.08)
            receiving_flow[arrived] = False
            for i in np.flatnonzero(arrived):
                if i not in filling and i not in yielding and i not in escapes:
                    targets[i] = []
                    fields.pop(i, None)
            if passage_count:
                receiving_flow[central_indices] |= side[central_indices] > 0
            for i, event in list(filling.items()):
                if np.linalg.norm(points[i] - event["goal_xy_m"]) < 0.035:
                    event["filled_route_step"] = step
                    filling.pop(i)
                    if event["chain_depth"] < 2:
                        vacancies.append(
                            {
                                "xy": np.asarray(event["origin_xy_m"]),
                                "visitor": event["visitor_index"],
                                "depth": event["chain_depth"] + 1,
                                "previous_mover": i,
                                "created_step": step,
                            }
                        )
                elif step - event["assigned_route_step"] > 110:
                    filling.pop(i)
            for i, event in list(yielding.items()):
                if np.linalg.norm(points[i] - event["goal_xy_m"]) < 0.025:
                    event["cleared_route_step"] = step
                    yielding.pop(i)
                elif step - event["assigned_route_step"] > 80:
                    yielding.pop(i)
            vacancy_goal(step)
            clear_entry(step)
            if pickup_passing:
                for i, event in list(passing_goals.items()):
                    if (np.linalg.norm(points[i]-event['goal']) < .025
                            or step-event['step'] > 240 or i in frozen):
                        passing_events.append(dict(actor_index=i, route_step=step,
                            event='finished' if np.linalg.norm(points[i]-event['goal']) < .025 else 'released',
                            blocker_index=event['blocker']))
                        passing_goals.pop(i)
                # Commit to the downstream side of a pickup rather than
                # repeatedly selecting new goals on the same blocked side.
                for i in central_indices:
                    if i in frozen or i in passing_goals or not receiving_flow[i]:
                        continue
                    for j in sorted(blockers):
                        ahead = (points[i, 1]-points[j, 1])*passage_direction
                        if (j == i or not .01 < ahead < .18
                                or abs(points[i, 0]-points[j, 0]) > separation+.035):
                            continue
                        goal = np.clip([points[i, 0], points[j, 1]-passage_direction*(separation+.07)],
                                       center_low+.004, center_high-.004)
                        if valid(goal, central=True):
                            passing_goals[i] = dict(goal=goal, blocker=j, step=step)
                            passing_events.append(dict(actor_index=i, blocker_index=j,
                                route_step=step, event='started', goal_xy_m=goal.tolist()))
                            break
                for i, event in passing_goals.items():
                    if i in frozen:
                        continue
                    goal = event['goal']
                    if not targets[i] or np.linalg.norm(targets[i][-1]-goal) > 1e-9:
                        targets[i] = navigation(points[i], goal, i)
                        fields.pop(i, None)
                    filling.pop(i, None)
                    yielding.pop(i, None)
                    escapes.pop(i, None)
            order = rng.permutation(count)
            local = np.zeros_like(points)
            limits = speed * speed_scale * (.65 + .35 * np.sin(step / 29 + np.arange(count)) ** 2)
            proposals = {}
            # First calculate independent local steering at the same time instant.
            # Then add the shared drift before selecting collision-safe moves.
            for index in order:
                if index in frozen:
                    continue
                if index in passage_set and loop_indices[index] in (1, 2):
                    last_gate = visitor_loops[index][loop_indices[index]-1]
                    if (before[index, 0] <= passage_x_max
                            and (before[index, 1]-last_gate[1])*passage_direction >= 0):
                        # Once a gate is passed, never steer back to its center
                        # just because a lateral avoidance move missed the point.
                        targets[index] = []
                        escapes.pop(index, None)
                        fields.pop(index, None)
                if index in escapes:
                    started, original = escapes[index]
                    if (np.linalg.norm(points[index]-targets[index][-1]) < .025
                            or step-started > 50):
                        targets[index] = original if index in passage_set else []
                        escapes.pop(index)
                        fields.pop(index, None)
                jammed = (step >= 30 and step-recovery_steps[index] >= 30
                          and np.linalg.norm(np.ptp(np.asarray(paths[-30:])[:, index], axis=0)) < .045)
                if jammed and index not in passing_goals:
                    recovery_steps[index] = step
                    if not escape(index, step):
                        # A neighboring cover can clear the route within a few
                        # frames. Failed recovery must not impose the full
                        # cooldown used after starting a successful detour.
                        recovery_steps[index] = step - 24
                        for other in np.argsort(np.linalg.norm(points-points[index], axis=1))[1:4]:
                            if other not in escapes and escape(int(other), step):
                                break
                if targets[index] and np.linalg.norm(points[index] - targets[index][-1]) < .025:
                    targets[index] = []
                while (
                    targets[index]
                    and np.linalg.norm(points[index] - targets[index][0]) < 0.018
                ):
                    targets[index].pop(0)
                if not targets[index] or (
                    index not in passage_set
                    and (step + (index * 23 if continuous else 0)) % (130 if continuous else 80) == 0
                    and index not in filling
                    and index not in yielding
                    and index not in escapes
                    and index not in passing_goals
                ):
                    if index in passage_set:
                        k = loop_indices[index]
                        goal = visitor_loops[index][k]
                        loop_indices[index] = (k + 1) % len(visitor_loops[index])
                    else:
                        goal = sample(index)
                    targets[index] = navigation(points[index], goal, index)
                velocity = limits[index]
                candidates = before[index] + velocity * units
                allowed = valid(candidates, central=index not in passage_set, index=index)
                allowed &= forward_crossing(before[index], candidates)
                if index not in passage_set:
                    allowed &= ~passage_intersects(before[index], candidates, passage_x_max, passage_y_range)
                costs = traffic_costs(index, candidates, step) - .0008 * (units @ headings[index])
                costs += crowding_cost(index, candidates)
                if (index in passage_set and loop_indices[index] == 2
                        and passage_intersects(before[index], before[index], passage_x_max, passage_y_range)):
                    # Follow the validated straight crossing once committed.
                    # A discretized detour field can otherwise favor sideways
                    # oscillation despite a clear forward lane.
                    costs = (-passage_direction * (candidates[:, 1]-before[index, 1])
                             + .7 * abs(candidates[:, 0]-rear_nodes[0, 0]))
                choices = np.flatnonzero(allowed)
                if len(choices):
                    local[index] = velocity * units[choices[np.argmin(costs[choices])]]
                proposals[index] = (candidates, costs)
            receiving = receiving_flow.copy()
            if continuous:
                # A temporary clearance/escape goal must control its own
                # direction. Shared drift otherwise pushes a yielding cover
                # back into the obstruction it is trying to leave.
                receiving[list(set(escapes) | set(yielding) | set(filling) | frozen)] = False
                if pickup_passing:
                    # Let the shortest-path field steer around a pickup.
                    # Shared drift can otherwise pin a neighbor against the
                    # stopped pair despite available space to its side.
                    receiving[list(bypass_indices | set(passing_goals))] = False
            preferred = receiving_side_velocities(local, receiving, passage_direction, limits)
            velocity_flow.append(dict(
                receiving_actor_indices=np.flatnonzero(receiving).tolist(),
                preferred_sum_toward_a_m_per_step=float(-passage_direction * preferred[receiving, 1].sum()),
                preferred_reverse_actor_count=int(np.sum(preferred[receiving, 1]*passage_direction > 0)),
            ))
            for index in order:
                if index in frozen:
                    velocities[index] = 0
                    continue
                candidates, costs = proposals[index]
                if continuous:
                    # Smooth velocity changes rather than jumping between
                    # independent angular samples on every control frame.
                    wanted = preferred[index]
                    previous = velocities[index]
                    desired = np.concatenate((
                        wanted[None],
                        limits[index]*units, .5*limits[index]*units,
                        .25*limits[index]*units, np.zeros((1, 2))))
                    acceleration = desired-previous
                    deltas = previous + acceleration*np.minimum(
                        1., .20*speed / np.maximum(np.linalg.norm(acceleration, axis=1), 1e-12))[:, None]
                    if speed_scale != 1.:
                        deltas *= np.minimum(1., limits[index] / np.maximum(
                            np.linalg.norm(deltas, axis=1), 1e-12))[:, None]
                    # A true emergency stop remains available if external
                    # contact or another stopped pair closes the local route.
                    deltas = np.r_[deltas, np.zeros((1, 2))]
                    candidates = before[index] + deltas
                    costs = (np.sum((deltas-wanted)**2, axis=1)
                             + 1.5*np.sum((deltas-previous)**2, axis=1)) / (2*limits[index])
                    # Look ahead at nearby moving traffic; favor early, gentle
                    # avoidance while retaining exact swept collision checks.
                    others = np.arange(count) != index
                    relative = before[index] - before[others]
                    rel_v = deltas[:, None] - velocities[others]
                    t = np.clip(-np.sum(relative[None]*rel_v, axis=2)
                                / np.maximum(np.sum(rel_v**2, axis=2), 1e-20), 0, 7)
                    gap = np.linalg.norm(relative[None] + t[..., None]*rel_v, axis=2)
                    costs += .02*np.maximum(pair_clearance(index)+.035-gap, 0).sum(axis=1)
                elif receiving[index]:
                    # Slower candidates and the exact desired velocity avoid
                    # rounding the group drift to a fixed-speed direction.
                    deltas = np.concatenate((limits[index]*units, .5*limits[index]*units,
                                             preferred[index:index+1]))
                    candidates = before[index] + deltas
                    costs = np.sum((deltas-preferred[index])**2, axis=1) / (2*limits[index])
                    # Occupied A-side space can redirect or slow the incoming
                    # flow. Never force the common velocity through a crowd.
                    costs += crowding_cost(index, candidates)
                allowed = valid(candidates, central=index not in passage_set, index=index)
                allowed &= forward_crossing(before[index], candidates)
                if index not in passage_set:
                    allowed &= ~passage_intersects(before[index], candidates, passage_x_max, passage_y_range)
                other_indices = np.arange(count) != index
                endpoint_clearance = pair_clearance(index)
                swept_clearance = pair_clearance(index, extra=0)
                if allow_contact_recovery:
                    # Contact or an externally replaced cover can reduce the
                    # reserved margin. Leave the real poses in place, allowing
                    # separation without making any existing overlap worse.
                    initial_distances = np.linalg.norm(before[index]-before[other_indices], axis=1)
                    endpoint_clearance = np.minimum(endpoint_clearance, initial_distances-1e-10)
                    swept_clearance = np.minimum(swept_clearance, initial_distances-1e-10)
                allowed &= np.all(np.linalg.norm(candidates[:, None] - points[other_indices], axis=2)
                                  > endpoint_clearance, axis=1)
                relative = before[index] - before[other_indices]
                velocity_delta = ((candidates - before[index])[:, None]
                                  - (points[other_indices] - before[other_indices]))
                t = np.clip(-np.sum(relative[None] * velocity_delta, axis=2)
                            / np.maximum(np.sum(velocity_delta**2, axis=2), 1e-20), 0, 1)
                allowed &= np.all(np.linalg.norm(relative[None] + t[..., None]*velocity_delta,
                                                 axis=2) >= swept_clearance, axis=1)
                # Empty covers placed aside remain obstacles after their cubes
                # rejoin traffic. Check the entire move, allowing escape from
                # an existing overlap without increasing the penetration.
                for center, parked_radius in parked:
                    relative_parked = before[index] - center
                    delta = candidates - before[index]
                    when = np.clip(-np.sum(relative_parked * delta, axis=1)
                                   / np.maximum(np.sum(delta**2, axis=1), 1e-20), 0, 1)
                    clearance = np.linalg.norm(relative_parked + when[:, None] * delta, axis=1)
                    required = min((actor_radii[index] if pickup_passing else radius) + parked_radius + MARGIN_M,
                                   float(np.linalg.norm(relative_parked)))
                    allowed &= clearance >= required - 1e-10
                choices = [k for k in np.flatnonzero(allowed)
                           if reachable_area.segment(before[index], candidates[k])]
                if choices:
                    k = choices[int(np.argmin(costs[choices]))]
                    delta = candidates[k] - before[index]
                    points[index] = candidates[k]
                    headings[index] = delta / max(np.linalg.norm(delta), 1e-12)
                velocities[index] = points[index] - before[index]
            for i in passage_indices:
                inside = bool(valid(points[i], central=True))
                if (not inside and i not in departed
                        and np.linalg.norm(points[i] - last_central[i]) > separation
                        and np.linalg.norm(points - last_central[i], axis=1).min() > separation):
                    vacancies.append(
                        {"xy": last_central[i].copy(), "visitor": i, "depth": 0,
                         "created_step": step}
                    )
                    departed.add(i)
                visitor_inside[i] = inside
                # Hysteresis: do not create new vacancies by oscillating along the boundary.
                if inside and (i not in departed or loop_indices[i] in (0, 4)):
                    last_central[i] = points[i].copy()
                    if loop_indices[i] in (0, 4):
                        departed.discard(i)
            gap = swept_gap(before, points, actor_radii)
            required_gap = MARGIN_M
            if allow_contact_recovery:
                required_gap = min(required_gap, swept_gap(before, before, actor_radii)) - 1e-9
            if gap < required_gap:
                raise AssertionError(f"Container footprints intersect: {gap}")
            minimum = min(minimum, gap)
            if pickup_bypass_margin or pickup_passing:
                outside_now = {i for i in range(count) if np.any(points[i] < low)
                               or np.any(points[i] > high)}
                for i in sorted(outside_now ^ outside_last):
                    bypass_events.append(dict(actor_index=i, route_step=step,
                        event='outside' if i in outside_now else 'returned', xy_m=points[i].tolist()))
                outside_last.clear()
                outside_last.update(outside_now)
            paths.append(points.copy())

    iterator = stream()
    first = next(iterator)
    if not continuous:
        for _ in range(steps):
            next(iterator)
        iterator.close()
    snapshot = np.asarray(paths)
    lengths = np.linalg.norm(np.diff(snapshot, axis=0), axis=2).sum(axis=0)
    if not continuous and np.min(lengths) < min(.15, steps * speed * .2):
        raise ValueError(f"A route stalled: {lengths}")
    metadata = {
        "motion_design": "original_passage_and_vacancy_flow_with_traffic_recovery",
        "actor_count": count,
        "radius_m": radius,
        "steps": steps,
        "seed": seed,
        "bounds_xy_m": bounds.tolist(),
        "central_bounds_xy_m": central_bounds.tolist(),
        "initial_bounds_xy_m": start_bounds.tolist(),
        "initial_center_bounds_xy_m": np.c_[start_bounds[:, 0] + radius,
                                              start_bounds[:, 1] - radius].tolist(),
        "central_actor_indices": central_indices,
        "central_route_exclusion": dict(
            start_xy_m=central_barrier_start[0].tolist(),
            end_xy_m=central_barrier_end[0].tolist(),
            center_clearance_m=allowance,
            physical_obstacle=False,
        ),
        "passage_actor_indices": passage_indices,
        "max_robot_side_visitors": passage_count,
        "passage_role_sampling": "nearest safe approaches to a uniformly seeded entry side; independent of RGB and goal assignment",
        "passage_direction_sampling": "one uniform left-to-right or right-to-left direction per episode; same direction in the narrow passage",
        "one_way_passage": dict(
            x_max_m=passage_x_max, y_range_m=passage_y_range.tolist(),
            direction=passage_direction, central_containers_excluded=True,
            scope="entire rear passage including entry and exit; navigation, detours, recovery, and executed moves",
        ),
        "passage_routes": [
            dict(actor_index=i,
                 direction="right-to-left" if directions[i] == 1 else "left-to-right",
                 waypoints_xy_m=visitor_loops[i].tolist())
            for i in passage_indices
        ],
        "max_speed_m_per_control_step": speed,
        "minimum_continuous_pair_gap_m": minimum,
        "route_lengths_m": lengths.tolist(),
        "traffic_direction": passage_direction,
        "receiving_side_flow": "shared B-to-A velocity on every step; local steering may reverse; collision safety has priority",
        "receiving_side_minimum_preferred_mean_speed_fraction": .25,
        "receiving_side_flow_release_distance_m": .08,
        "central_crowding_cost_weight_m": .045,
        "receiving_side_velocity_steps": velocity_flow,
        "traffic_recovery_events": recovery_events,
        "strategy": f"{passage_count} covers circulate around the buttons; {len(central_indices)} central covers follow vacated spaces and clear returning visitors; only button gap closed",
        "initial_packing_attempts": packing_attempts,
        "vacancy_flow_events": flow_events,
        "passage_loop_waypoints_xy_m": visitor_loop.tolist(),
        "robot_side_crossing_xy_m": rear_nodes.tolist(),
        "robot_side_crossing_entire_segment_reachable": True,
        "robot_base_keepout_radius_m": ROBOT_BASE_RADIUS_M,
        "obstacles": [{"xy": p.tolist(), "radius_m": r} for p, r in obstacles],
        "rectangle_obstacles": [{"xy_m": p.tolist(), "half_size_xy_m": h.tolist()}
                                for p, h in zip(box_centers, box_half_sizes)],
        "route_exclusions": {
            "starts_xy_m": starts.tolist(),
            "ends_xy_m": ends.tolist(),
            "radii_m": widths.tolist(),
            "physical_obstacle": False,
            "between_buttons_allowed": False,
            "far_side_crossing_allowed": True,
        },
    }

    if continuous:
        metadata.update(motion_design="continuous_rotating_visitors_smooth_flow", steps=None,
                        minimum_continuous_pair_gap_m=None,
                        passage_assignments=passage_assignments,
                        active_passage_actor_indices=active_passages,
                        passage_actor_indices=list(range(count)),
                        central_actor_indices=list(range(count)),
                        passage_routes=[dict(actor_index=i,
                            direction='right-to-left' if passage_direction == 1 else 'left-to-right',
                            waypoints_xy_m=(visitor_loop if passage_direction == 1
                                            else visitor_loop[[1, 0, 4, 3, 2]]).tolist()) for i in range(count)],
                        strategy='New visitors take over the one-way button passage after each exit; all covers are eligible',
                        central_passage_clearance_m=lane_clearance,
                        velocity_change_limit_m_per_step_squared=.20*speed,
                        emergency_stop_allowed=True,
                        passage_role_sampling='available covers near the entry, with seeded variation and least-used preference')
        if pickup_bypass_margin or pickup_passing:
            metadata.update(pickup_bypass_margin_m=pickup_bypass_margin,
                            pickup_bypass_bounds_xy_m=np.c_[bounds[:, 0]-pickup_bypass_margin,
                                                           bounds[:, 1]+pickup_bypass_margin].tolist(),
                            pickup_bypass_trigger_radius_m=.22,
                            pickup_passing=pickup_passing,
                            pickup_passing_events=passing_events,
                            traffic_grid_spacing_m=grid_spacing,
                            pickup_traffic_grid_spacing_m=.006 if pickup_passing else grid_spacing,
                            pickup_bypass_events=bypass_events,
                            pickup_bypass_policy='Nearby traffic may detour during pickup; destinations remain inside the nominal bounds; reachability and obstacle clearance remain enforced')
        return (first, iterator), metadata
    return snapshot, metadata


def packed_initial(count, radius, seed, speed, obstacles, reject=None, reachable_area=None,
                   extra_obstacles=(), bounds=None, extra_spacing=None):
    """Pack irregular layouts without shrinking covers or changing clearance."""
    rng = np.random.default_rng(seed + 16000)
    starts, ends, widths, _ = barriers(obstacles)
    circles = [*obstacles, *extra_obstacles, (ROBOT_BASE_XY, ROBOT_BASE_RADIUS_M)]
    centers, sizes = (
        np.array([p for p, _ in circles]),
        np.array([r for _, r in circles]),
    )
    bounds = initial_bounds(obstacles) if bounds is None else np.asarray(bounds)
    low, high = bounds[:, 0] + radius, bounds[:, 1] - radius
    if np.any(high <= low):
        raise ValueError("No initial area between the buttons and board")
    allowance = radius + MARGIN_M + speed
    # Online traffic checks swept clearance jointly on every step. Compact
    # layouts need not reserve two additional full-speed steps per cover.
    separation = 2 * radius + MARGIN_M + (2 * speed if extra_spacing is None else extra_spacing)
    pool = rng.uniform(low, high, (12000, 2))
    valid = np.all(
        point_segment_distance(pool, starts, ends) > widths + allowance, axis=1
    ) & np.all(
        np.linalg.norm(pool[:, None] - centers, axis=2) > sizes + allowance, axis=1
    )
    pool = pool[valid]
    if reachable_area is not None:
        pool = pool[reachable_area.contains(pool)]
    for attempt in range(256):
        points = []
        nearest = np.full(len(pool), np.inf)
        for index in range(count):
            choices = np.flatnonzero(nearest > separation)
            if reject:
                choices = np.array(
                    [k for k in choices if not reject(index, pool[k])], dtype=int
                )
            if not len(choices):
                break
            p = pool[int(rng.choice(choices))]
            points.append(p.copy())
            nearest = np.minimum(nearest, np.linalg.norm(pool - p, axis=1))
        if len(points) == count:
            points = np.asarray(points)[rng.permutation(count)]
            if reject is not None and any(reject(i, p) for i, p in enumerate(points)):
                continue
            return points, attempt + 1
    raise ValueError(f"Cannot fit {count} containers with the existing clearance")
