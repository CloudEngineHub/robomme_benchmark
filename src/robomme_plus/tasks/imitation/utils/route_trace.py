"""Per-curve white-dot limits in coordinates defined by consecutive nodes."""

import numpy as np


def project_dots(points_xy, start_xy, end_xy):
    """Return longitudinal position, absolute perpendicular distance and span, in meters."""
    start, end = np.asarray(start_xy, float), np.asarray(end_xy, float)
    span = float(np.linalg.norm(end - start))
    if not np.isfinite(span) or span <= 1e-9:
        raise ValueError("Route curve endpoints must be distinct finite points")
    axis = (end - start) / span
    delta = np.asarray(points_xy, float).reshape(-1, 2) - start
    return delta @ axis, np.abs(delta[:, 0] * axis[1] - delta[:, 1] * axis[0]), span


class RouteTrace:
    """Bound each curve's bulge by its own demonstration and endpoint overshoot.

    No nearest-curve or timing match is required. Existing target-order,
    direction, contact and drawing-continuity rules remain in force.
    """

    def __init__(self, nodes_xy, lateral_ratio=1.3, endpoint_margin_m=.03):
        self.demo_nodes = np.asarray(nodes_xy, float).reshape(-1, 2).copy()
        if len(self.demo_nodes) < 2 or not np.isfinite(self.demo_nodes).all():
            raise ValueError("Route trace requires finite consecutive node coordinates")
        for start, end in zip(self.demo_nodes, self.demo_nodes[1:]):
            project_dots([], start, end)
        self.execution_nodes = None
        self.lateral_ratio = float(lateral_ratio)
        self.endpoint_margin_m = float(endpoint_margin_m)
        if not np.isfinite(lateral_ratio) or lateral_ratio < 1:
            raise ValueError("The lateral ratio must be finite and at least one")
        if not np.isfinite(endpoint_margin_m) or endpoint_margin_m < 0:
            raise ValueError("The endpoint margin must be finite and nonnegative")
        count = len(self.demo_nodes) - 1
        self.points = {phase: [[] for _ in range(count)] for phase in ("demonstration", "execution")}
        self.oracle_extents = None
        self.seen = set()
        self.failure = None
        self.finalized = False

    def begin_execution(self, nodes_xy):
        self.execution_nodes = np.asarray(nodes_xy, float).reshape(-1, 2).copy()
        if self.execution_nodes.shape != self.demo_nodes.shape or not np.isfinite(self.execution_nodes).all():
            raise ValueError("Execution must preserve ordered route node identities")
        self.oracle_extents = []
        for index, dots in enumerate(self.points["demonstration"]):
            _, lateral, demo_span = project_dots(dots, *self.demo_nodes[index:index + 2])
            _, _, execution_span = project_dots([], *self.execution_nodes[index:index + 2])
            # The existing row transform is a uniform planar similarity.
            # Translation and rotation preserve distances; spacing scales them.
            self.oracle_extents.append(float(lateral.max()) * execution_span / demo_span if len(lateral) else None)

    def update(self, phase, curve, step, xy, draws=True):
        if (self.failure is not None or not draws or curve is None
                or phase not in self.points or (phase, int(step)) in self.seen):
            return
        if not 0 <= curve < len(self.points[phase]):
            raise ValueError("Invalid route curve index")
        xy = np.asarray(xy, float).reshape(2)
        if not np.isfinite(xy).all():
            return
        self.seen.add((phase, int(step)))
        self.points[phase][curve].append(xy.tolist())
        if phase != "execution":
            return
        reference = None if self.oracle_extents is None else self.oracle_extents[curve]
        if reference is None:
            self.failure = dict(reason="trace_reference_missing", curve=int(curve), step=int(step))
            return
        along, lateral, span = project_dots([xy], *self.execution_nodes[curve:curve + 2])
        overshoot = max(0., -float(along[0]), float(along[0]) - span)
        if overshoot > self.endpoint_margin_m + 1e-9:
            self.failure = dict(reason="trace_endpoint_overshoot", curve=int(curve), step=int(step),
                                overshoot_m=overshoot, limit_m=self.endpoint_margin_m)
        elif lateral[0] > reference * self.lateral_ratio + 1e-9:
            self.failure = dict(reason="trace_lateral_extent", curve=int(curve), step=int(step),
                                distance_m=float(lateral[0]), oracle_distance_m=reference,
                                limit_m=reference * self.lateral_ratio)

    def finish(self, step):
        if self.failure is None:
            for index, points in enumerate(self.points["execution"]):
                if not points:
                    self.failure = dict(reason="trace_incomplete", curve=index, step=int(step))
                    break
        self.finalized = True

    def summary(self, include_points=False):
        curves = []
        for index, points in enumerate(self.points["execution"]):
            nodes = self.demo_nodes if self.execution_nodes is None else self.execution_nodes
            along, lateral, span = project_dots(points, *nodes[index:index + 2])
            oracle = None if self.oracle_extents is None else self.oracle_extents[index]
            maximum = float(lateral.max()) if len(lateral) else None
            row = dict(curve=index, start_xy_m=nodes[index].tolist(), end_xy_m=nodes[index + 1].tolist(),
                       oracle_max_distance_m=oracle, lateral_limit_m=None if oracle is None else oracle * self.lateral_ratio,
                       execution_max_distance_m=maximum,
                       lateral_ratio=None if maximum is None or not oracle else maximum / oracle,
                       projection_min_m=float(along.min()) if len(along) else None,
                       projection_max_m=float(along.max()) if len(along) else None, node_span_m=span,
                       projection_overshoot_m=max(0., -float(along.min()), float(along.max()) - span) if len(along) else None,
                       drawing_points=len(points))
            if include_points:
                row.update(demonstration_xy_m=self.points["demonstration"][index], execution_xy_m=points)
            curves.append(row)
        return dict(lateral_ratio_limit=self.lateral_ratio, endpoint_margin_m=self.endpoint_margin_m,
                    finalized=self.finalized, failure=self.failure, curves=curves)
