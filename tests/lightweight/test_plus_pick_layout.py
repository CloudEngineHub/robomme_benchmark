"""PickXtimes samples the trajectory before using the remaining reachable table."""
import unittest

import numpy as np

from robomme_plus.tasks.counting.utils.motion import (
    PICK_BUTTON_HALF_SIZE,
    PICK_BUTTON_RADIUS,
    PICK_LAYOUT_GAP,
    PICK_MAX_MANIPULATION_REACH,
    PICK_MIN_MANIPULATION_REACH,
    PICK_OBJECT_BOUNDS,
    PICK_ROBOT_BASE_XY,
    TARGET_DISC_RADIUS,
    seeded_counting_layout,
)


def layout(seed, distractors=3, **kwargs):
    return seeded_counting_layout('PickXtimes', seed, distractors=distractors,
                                  ellipse_scale=(1., 1.2), **kwargs)


class PickLayoutTest(unittest.TestCase):
    def test_replay_and_route_button_independent_of_cube_population(self):
        for seed in range(32):
            a, b = layout(seed, 0), layout(seed, 5)
            self.assertEqual(a, layout(seed, 0))
            for key in ('ellipse_center_xyz','ellipse_axes_m','ellipse_scale',
                        'initial_phase_rad','button_xy'):
                self.assertEqual(a[key], b[key], (seed, key))

    def test_every_footprint_clears_the_entire_moving_path(self):
        theta = np.linspace(0, 2*np.pi, 8192, endpoint=False)
        for seed in range(64):
            s = layout(seed)
            path = np.array(s['ellipse_center_xyz'][:2]) + np.array(s['ellipse_axes_m'])*np.stack([np.cos(theta),np.sin(theta)],axis=1)
            button_gap = np.linalg.norm(
                np.maximum(np.abs(path-s['button_xy'])-PICK_BUTTON_HALF_SIZE,0),axis=1
            ).min()
            self.assertGreaterEqual(button_gap,TARGET_DISC_RADIUS+PICK_LAYOUT_GAP)
            points = [s['cube_xyz'], *s['distractor_xyz']]
            yaws = [s['cube_yaw_rad'], *s['distractor_yaw_rad']]
            occupied = [(np.array(s['button_xy']), PICK_BUTTON_RADIUS)]
            for point, yaw in zip(points,yaws):
                xy = np.array(point[:2])
                reach = np.linalg.norm(xy-PICK_ROBOT_BASE_XY)
                self.assertGreaterEqual(reach,PICK_MIN_MANIPULATION_REACH)
                self.assertLessEqual(reach,PICK_MAX_MANIPULATION_REACH)
                self.assertTrue(np.all(xy>=PICK_OBJECT_BOUNDS[0]))
                self.assertTrue(np.all(xy<=PICK_OBJECT_BOUNDS[1]))
                c, sn = np.cos(yaw), np.sin(yaw)
                local = (path-xy) @ np.array([[c,-sn],[sn,c]])
                gap = np.linalg.norm(np.maximum(np.abs(local)-.02,0),axis=1).min()
                self.assertGreaterEqual(gap,TARGET_DISC_RADIUS+PICK_LAYOUT_GAP)
                for other,radius in occupied:
                    self.assertGreaterEqual(np.linalg.norm(xy-other),np.sqrt(2)*.02+radius+PICK_LAYOUT_GAP)
                occupied.append((xy,np.sqrt(2)*.02))

    def test_wide_coverage_random_yaws_and_cubes_inside_the_loop(self):
        samples = [layout(seed) for seed in range(512)]
        chosen = np.array([s['cube_xyz'][:2] for s in samples])
        self.assertGreater(np.ptp(chosen[:,0]),.45)
        self.assertGreater(np.ptp(chosen[:,1]),.60)
        self.assertTrue(np.any(np.abs(chosen[:,1])<.02))
        yaws = np.array([s['cube_yaw_rad'] for s in samples])
        self.assertGreater(np.ptp(yaws),6.0)
        self.assertTrue(np.all(np.histogram(yaws,bins=np.linspace(-np.pi,np.pi,13))[0]>20))
        inside = []
        for s in samples:
            positions = np.array([s['cube_xyz'][:2],*[p[:2] for p in s['distractor_xyz']]])
            centered = (positions-np.array(s['ellipse_center_xyz'][:2]))/s['ellipse_axes_m']
            inside.extend(np.sum(centered**2,axis=1)<1)
        self.assertTrue(any(inside), 'Interior must remain usable when the rotated cube fits')
        self.assertTrue(not all(inside))


if __name__=='__main__':
    unittest.main()
