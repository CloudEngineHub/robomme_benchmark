"""Long line routes and the hard-only pair of touching ellipses."""

import unittest

import numpy as np

from robomme_plus.tasks.counting.utils.controllers import PROFILES, resolve_counting_config
from robomme_plus.tasks.counting.utils.motion import (
    CurvedRoute, STOP_PATH_BOUNDS, seeded_stop_layout,
)
from robomme_plus.tasks.counting.utils.variations import (
    SHAPES, STOP_DURATIONS, STOP_SHAPES, sample_variation,
)


def config(seed, level):
    return resolve_counting_config(
        dict(PROFILES["stopcube_random"], profile="stopcube_random", level=level), seed
    )


def route_from(config):
    return CurvedRoute(
        config["shape"], config["durations"], axes=config["axes"],
        rotation_degrees=config["rotation_degrees"],
        direction_sign=config.get("direction_sign", 1),
    )


class StopRoutesTest(unittest.TestCase):
    def test_new_shape_is_hard_only_and_seeded(self):
        for level in ("easy", "medium", "hard"):
            shapes = [sample_variation("StopCube", seed, level)["shape"] for seed in range(1000)]
            self.assertEqual(set(shapes), set(STOP_SHAPES[level]))
            self.assertEqual(set(shapes) - set(SHAPES), {"double_ellipse"} if level == "hard" else set())
            for shape in STOP_SHAPES[level]:
                self.assertGreater(shapes.count(shape), 140)
                self.assertLess(shapes.count(shape), 330)
            self.assertEqual(shapes[93], sample_variation("StopCube", 93, level)["shape"])

    def test_medium_hard_lines_are_longer_and_keep_the_speed_schedule(self):
        for level in ("medium", "hard"):
            lengths = []
            for seed in range(256):
                cfg = config(seed, level)
                if cfg["shape"] != "line":
                    continue
                length = 2 * cfg["axes"][0]
                lengths.append(length)
                self.assertGreaterEqual(length, .45)
                self.assertLessEqual(length, .52)
                # Length and travel time change together; only integer-step
                # rounding may perturb the original segment speeds.
                nominal = np.asarray(STOP_DURATIONS[cfg["speed"]]) * length / .30
                np.testing.assert_allclose(cfg["durations"], nominal, atol=.5, rtol=0)
                route = route_from(cfg)
                layout = seeded_stop_layout(seed, route, level)
                outline = route.local_points(np.linspace(0, 2, 1025)) + layout["route_center_xy"]
                self.assertTrue(np.all(outline >= STOP_PATH_BOUNDS[0]))
                self.assertTrue(np.all(outline <= STOP_PATH_BOUNDS[1]))
            self.assertGreater(max(lengths) - min(lengths), .06)

    def test_easy_line_length_is_unchanged(self):
        for seed in range(64):
            cfg = config(seed, "easy")
            if cfg["shape"] == "line":
                scale = np.random.default_rng([seed, 271828, 3]).uniform(.7, 1.3)
                self.assertAlmostEqual(2 * cfg["axes"][0], .30 * scale)

    def test_two_exact_ellipses_touch_at_long_side_midpoints(self):
        a, b = .14, .045
        route = CurvedRoute("double_ellipse", [40], axes=(a, b), center=(0, 0))
        points = route.local_points(np.linspace(0, 2, 2049))
        ellipse_equation = (points[:, 0] / a)**2 + ((np.abs(points[:, 1]) - b) / b)**2
        np.testing.assert_allclose(ellipse_equation, 1, atol=1e-12)
        np.testing.assert_allclose(route.target, [0, 0])
        np.testing.assert_allclose(route.local_points([0, .5, 1, 1.5, 2]),
                                   [[0, 2*b], [0, 0], [0, -2*b], [0, 0], [0, 2*b]], atol=1e-12)

    def test_new_curve_matches_original_figure_eight_mean_speeds(self):
        progress = np.linspace(0, 2, 16385)
        original = CurvedRoute("figure8", [40]).local_points(progress)
        original_length = np.linalg.norm(np.diff(original, axis=0), axis=1).sum()
        for seed in range(64):
            cfg = config(seed, "hard")
            if cfg["shape"] != "double_ellipse":
                continue
            points = route_from(cfg).local_points(progress)
            length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            nominal = np.asarray(STOP_DURATIONS[cfg["speed"]]) * length / original_length
            np.testing.assert_allclose(cfg["durations"], nominal, atol=.501, rtol=0)

    def test_join_is_continuous_with_a_shared_tangent_in_both_directions(self):
        epsilon = 1e-7
        for direction in (-1, 1):
            route = CurvedRoute("double_ellipse", [40], direction_sign=direction)
            for crossing in (.5, 1.5, 2.5):
                center = route.local_points(crossing)
                before = (center - route.local_points(crossing - epsilon)) / epsilon
                after = (route.local_points(crossing + epsilon) - center) / epsilon
                np.testing.assert_allclose(before, after, atol=1e-6)
                self.assertGreater(direction * before[0], 0)

    def test_new_shape_fits_and_counts_once_per_crossing_at_control_rate(self):
        seen = set()
        for seed in range(128):
            cfg = config(seed, "hard")
            if cfg["shape"] != "double_ellipse":
                continue
            route = route_from(cfg)
            scene = seeded_stop_layout(seed, route, "hard")
            route.center = tuple(scene["route_center_xy"])
            seen.add((cfg["speed"], cfg["direction_sign"]))
            outline = route.local_points(np.linspace(0, 2, 4097)) + route.center
            self.assertTrue(np.all(outline >= STOP_PATH_BOUNDS[0]))
            self.assertTrue(np.all(outline <= STOP_PATH_BOUNDS[1]))
            for visit in range(1, 13):
                np.testing.assert_allclose(route.point(route.crossing_step(visit)), route.target, atol=1e-12)
            visits, inside = 0, False
            for step in range(int(np.ceil(route.crossing_step(12))) + 1):
                distance = np.linalg.norm(route.point(step) - route.target)
                near = distance <= (.04 if inside else .032)
                visits += bool(near and not inside)
                inside = near
            self.assertEqual(visits, 12, (seed, cfg))
        self.assertEqual(seen, {(speed, direction) for speed in ("mixed_loops", "mixed_halves") for direction in (-1, 1)})


if __name__ == "__main__":
    unittest.main()
