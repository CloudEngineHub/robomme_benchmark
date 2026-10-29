"""Intermittent blockers must be replayable, retract, and preserve drawing rules."""
import unittest

import numpy as np

from robomme_plus.evaluation.metadata import normalize_episode
from robomme_plus.tasks.imitation.utils.drawing import Drawing
from robomme_plus.tasks.imitation.utils.route_crossing import crossing_positions, crossing_state
from robomme_plus.tasks.imitation.utils.route_layout import resolve_route_layout
from robomme_plus.tasks.imitation.utils.route_motion import resolve_board_motion


class CrossingTest(unittest.TestCase):
    def config(self, **kwargs):
        return resolve_board_motion(0,dict(mode="crossing_boards",start_delays_s=[0,0,0,0],**kwargs))

    def test_clear_crossing_and_retraction_stages(self):
        p=self.config(probability=1.)
        for t,extent,stage in ((0,0,"entering"),(.75,.1,"blocking"),(2.25,.1,"retracting"),
                                (3,0,"clear"),(5.9,0,"clear")):
            offsets,states=crossing_state(p,t)
            np.testing.assert_allclose(np.abs(offsets),extent,atol=1e-12)
            self.assertTrue(all(s["stage"]==stage for s in states))
        for boundary in (0.,.75,2.25,3.,6.):
            before,_=crossing_state(p,boundary-1e-5)
            after,_=crossing_state(p,boundary+1e-5)
            self.assertLess(np.max(np.abs(after-before)),1e-8)

    def test_seeded_decisions_do_not_depend_on_query_order_and_include_skips(self):
        p=self.config()
        times=np.linspace(0,120,401)
        original=[crossing_state(p,t) for t in times]
        for i in np.random.default_rng(42).permutation(len(times)):
            offsets,states=crossing_state(p,times[i])
            np.testing.assert_array_equal(offsets,original[i][0])
            self.assertEqual(states,original[i][1])
        choices={s["chosen"] for _,states in original for s in states}
        self.assertEqual(choices,{True,False})
        self.assertEqual({s["side"] for _,states in original for s in states},{-1,1})
        for t in times:
            np.testing.assert_array_equal(crossing_state(self.config(probability=0),t)[0],np.zeros(4))

    def test_motion_is_perpendicular_and_follows_execution_spacing(self):
        rest=np.array([[.1,y] for y in (-.24,-.08,.08,.24)])
        p=self.config(probability=1)
        for angle in (0,.7,-1.2):
            r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            row=np.array([0,1])@r.T
            rotated=rest@r.T
            points,_=crossing_positions(rotated,row,p,1.,.9)
            delta=points-rotated
            np.testing.assert_allclose(delta@row,0,atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(delta,axis=1),.09)

    def test_metadata_roundtrip_and_invalid_parameters(self):
        layout=resolve_route_layout(3,dict(version=3,board_motion=dict(mode="crossing_boards")))
        row=normalize_episode("RouteStick",dict(seed=3,difficulty="hard",oracle_policy_step=100,
                              execution_route_layout=layout),0)
        self.assertEqual(normalize_episode("RouteStick",row,0),row)
        for bad in (dict(period_s=2),dict(move_s=0),dict(hold_s=-1),dict(stroke_m=.2),
                    dict(probability=1.1),dict(probability=True),dict(schedule_seed=-1),
                    dict(start_delays_s=[0,1]),dict(phases=[]),dict(other=0)):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                resolve_board_motion(0,dict(mode="crossing_boards",**bad))

    def test_wait_or_lift_and_return_preserves_curve_continuity(self):
        for lift in (False,True):
            drawing=Drawing(); drawing.begin("execution",0)
            drawing.update(1,[.08,.10,.07])
            for step in range(2,62):
                drawing.update(step,[.02,.16,.20] if lift else [.08,.10,.07])
            drawing.update(62,[.08,.10,.07])
            self.assertIsNone(drawing.failure)
        drawing.update(63,[.08,.10,.20])
        drawing.update(64,[.12,.10,.07])
        self.assertEqual(drawing.failure["reason"],"discontinuous")


if __name__=="__main__":
    unittest.main()
