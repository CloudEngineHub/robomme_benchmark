"""Departure gates target the current curve and give a bounded reaction window."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robomme_plus.evaluation.metadata import normalize_episode
from robomme_plus.tasks.imitation.utils.route_departure import DepartureCrossings, interception_contact
from robomme_plus.tasks.imitation.utils.drawing import Drawing
from robomme_plus.tasks.imitation.utils.route_layout import execution_points, resolve_route_layout, rotation
from robomme_plus.tasks.imitation.utils.route_motion import resolve_board_motion


class DepartureTest(unittest.TestCase):
    def gate(self):
        return DepartureCrossings(resolve_board_motion(0,dict(mode="departure_boards")))

    def update(self, gate, point, time, **overrides):
        args=dict(curve=0,start_index=0,end_index=2,direction="clockwise",tip_xy=point,
                  start_xy=[0,0],drawing=True,time_s=time)
        args.update(overrides)
        gate.update(**args)

    def test_must_touch_start_then_leave_and_cannot_retrigger_on_jitter(self):
        g=self.gate()
        self.update(g,[.2,.2],0)
        self.assertFalse(g.events)
        self.update(g,[0,0],1,drawing=False)
        self.update(g,[.04,0],2,drawing=False)
        self.assertFalse(g.events)
        self.update(g,[0,0],3)
        self.update(g,[.034,0],4)
        self.assertFalse(g.events)
        self.update(g,[.036,0],5,drawing=False)
        self.assertEqual(len(g.events),1)
        for time,point in ((6,[0,0]),(7,[.04,0]),(20,[.1,0])):
            self.update(g,point,time)
        self.assertEqual(len(g.events),1)
        np.testing.assert_array_equal(g.offsets(100)[0],np.zeros(4))
        self.assertFalse(self.gate().events)

    def test_only_intervening_board_moves_toward_prescribed_curve_side(self):
        for start,end in ((0,2),(2,0),(4,6),(8,6)):
            for direction in ("clockwise","counterclockwise"):
                g=self.gate()
                kwargs=dict(start_index=start,end_index=end,direction=direction)
                self.update(g,[0,0],0,**kwargs)
                self.update(g,[.04,0],1,**kwargs)
                offsets,_=g.offsets(2.5)
                board=min(start,end)//2
                sign=(1 if end>start else -1)*(1 if direction=="clockwise" else -1)
                expected=np.zeros(4); expected[board]=sign*.1
                np.testing.assert_allclose(offsets,expected)

    def test_slow_smooth_entry_hold_return_and_no_teleport_on_early_backtrack(self):
        g=self.gate(); self.update(g,[0,0],0); self.update(g,[.04,0],1)
        times=np.linspace(1,5.5,901)
        offsets=np.array([g.offsets(t)[0] for t in times])
        self.assertLessEqual(np.abs(np.diff(offsets,axis=0)).max()/(times[1]-times[0]),.100001)
        np.testing.assert_array_equal(offsets[0],np.zeros(4))
        np.testing.assert_allclose(offsets[-1],0,atol=1e-12)
        before=g.offsets(2)[0]
        self.update(g,[0,0],1.9,curve=1,start_index=2,end_index=0)
        self.update(g,[.04,0],2,curve=1,start_index=2,end_index=0)
        np.testing.assert_array_equal(g.offsets(2)[0],before)
        self.assertEqual(len(g.events),1)
        self.update(g,[0,0],6,curve=2)
        self.update(g,[.04,0],7,curve=2)
        self.assertEqual(len(g.events),2)

    def test_uninterrupted_prescribed_tip_path_is_intercepted_after_departure(self):
        g=self.gate()
        hits=[]
        for i,u in enumerate(np.linspace(0,1,45)):
            time=i*.05
            point=np.array([-.4*u*(1-u),.16*u])
            self.update(g,point,time)
            offsets,_=g.offsets(time)
            board_center=np.array([-offsets[0],.08])
            delta=np.abs(point-board_center)-[.03,.015]
            # Distance from a radius-8-mm vertical stick to the board footprint.
            clearance=np.linalg.norm(np.maximum(delta,0))-.008
            if clearance<0:
                hits.append(time)
        self.assertTrue(hits)
        self.assertGreater(min(hits)-g.events[0]["started_s"],.5)

    def test_metadata_roundtrip_and_validation(self):
        layout=resolve_route_layout(0,dict(version=3,board_motion=dict(mode="departure_boards")))
        self.assertEqual(layout["board_motion"]["phases"],["demonstration","execution"])
        row=normalize_episode("RouteStick",dict(seed=0,difficulty="hard",oracle_policy_step=100,
                                               execution_route_layout=layout),0)
        self.assertEqual(normalize_episode("RouteStick",row,0),row)
        # Explicit recorded phases retain the earlier moving-demonstration experiment.
        saved=dict(layout["board_motion"],phases=["demonstration","execution"])
        self.assertEqual(resolve_board_motion(0,saved),saved)
        for bad in (dict(move_s=.5),dict(hold_s=-1),dict(return_s=0),dict(stroke_m=.2),
                    dict(departure_radius_m=.02),dict(arm_radius_m=True),dict(phases=[]),
                    dict(probability=.5)):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                resolve_board_motion(0,dict(mode="departure_boards",**bad))

    def test_execution_shift_rotation_and_spacing_preserve_interception(self):
        targets=np.array([[0.,.08*i] for i in range(9)])
        center=targets.mean(axis=0)
        for scale in (.9,1.):
            for yaw in (-10.,10.):
                for side in (-1,1):
                    with self.subTest(scale=scale,yaw=yaw,side=side):
                        layout=resolve_route_layout(0,dict(version=3,spacing_scale=scale,
                            yaw_deg=yaw,translation_xy_m=[.05,0.]))
                        nodes=execution_points(targets,center,layout)
                        rest=nodes[[1,3,5,7]]
                        row=nodes[-1]-nodes[0]
                        gate=self.gate()
                        hits=[]
                        for i,u in enumerate(np.linspace(0,1,45)):
                            time=i*.05
                            point=execution_points(np.array([side*.4*u*(1-u),.16*u]),center,layout)
                            gate.update(curve=0,start_index=0,end_index=2,
                                direction="clockwise" if side<0 else "counterclockwise",
                                tip_xy=point,start_xy=nodes[0],drawing=True,time_s=time)
                            boards,_=gate.positions(rest,row,time,scale)
                            np.testing.assert_allclose(boards[1:],rest[1:],atol=1e-12)
                            # Board dimensions stay fixed while the row and travel scale.
                            delta=np.abs((point-boards[0])@rotation(layout))-[.03,.015]
                            if np.linalg.norm(np.maximum(delta,0))<.008:
                                hits.append(time)
                        self.assertEqual(len(gate.events),1)
                        self.assertTrue(hits)
                        self.assertGreater(min(hits)-gate.events[0]["started_s"],.5)
                        peak,_=gate.positions(rest,row,gate.events[0]["started_s"]+1.5,scale)
                        self.assertAlmostEqual(np.linalg.norm(peak[0]-rest[0]),.1*scale)


class ArrivalTest(unittest.TestCase):
    def test_only_selected_boards_move_and_revisits_keep_their_choice(self):
        for mask in ([False] * 4, [True] * 4, [False, True, False, True]):
            parameters = resolve_board_motion(0, dict(mode="arrival_boards", enabled_boards=mask))
            gate = DepartureCrossings(parameters)
            for visit in range(2):
                for board in range(4):
                    curve = visit * 4 + board
                    time = curve * 6.
                    start, end = (2 * board, 2 * board + 2)
                    if visit:
                        start, end = end, start
                    args = dict(curve=curve, start_index=start, end_index=end,
                                direction="clockwise", tip_xy=[0, 0], start_xy=[0, 0], drawing=True)
                    gate.update(**args, time_s=time)
                    # Repeated physics callbacks must not reroll or retrigger.
                    gate.update(**args, time_s=time + .1)
                    offsets, _ = gate.offsets(time + 2.)
                    expected = np.zeros(4)
                    expected[board] = (.07 if visit == 0 else -.07) if mask[board] else 0.
                    np.testing.assert_allclose(offsets, expected)
            self.assertEqual(len(gate.events), 2 * sum(mask))
            self.assertEqual(gate.fired, set(range(8)))
            # Phase resets clear events but preserve which boards can move.
            reset = DepartureCrossings(parameters)
            self.assertFalse(reset.events)
            self.assertEqual(reset.enabled_boards, tuple(mask))

    def test_board_mask_metadata_roundtrip_and_validation(self):
        import json

        for mask in ([False] * 4, [True] * 4, [False, True, True, False]):
            layout = resolve_route_layout(3, dict(version=3, board_motion=dict(
                mode="arrival_boards", enabled_boards=mask)))
            row = normalize_episode("RouteStick", dict(seed=3, difficulty="hard",
                oracle_policy_step=100, execution_route_layout=layout), 0)
            restored = normalize_episode("RouteStick", json.loads(json.dumps(row)), 0)
            self.assertEqual(restored, row)
            motion = restored["execution_route_layout"]["board_motion"]
            self.assertEqual(motion["enabled_boards"], mask)
            self.assertEqual(resolve_board_motion(999, motion), motion)
        legacy = resolve_board_motion(3, dict(mode="arrival_boards"))
        self.assertNotIn("enabled_boards", legacy)
        self.assertEqual(DepartureCrossings(legacy).enabled_boards, (True,) * 4)
        for mask in (None, True, [], [True] * 3, [False] * 5, [True, False, 1, 0],
                     [True, False, "false", None], {0: True, 1: False, 2: True, 3: False}):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                resolve_board_motion(0, dict(mode="arrival_boards", enabled_boards=mask))

    def test_straight_stick_above_node_is_not_contact(self):
        env=SimpleNamespace(_drawing=Drawing(max_angle_deg=None))
        with patch('robomme_plus.tasks.imitation.utils.split_stick.tip_position') as position, \
                patch('robomme_plus.tasks.imitation.utils.split_stick.tip_angle_deg') as angle:
            for height,expected in ((.245,False),(.081,False),(.078,True),(-.001,False)):
                position.return_value=np.array([0.,0.,height])
                tip,contact=interception_contact(env)
                self.assertEqual(contact,expected)
                self.assertEqual(tip[2],height)
            angle.assert_not_called()
            env._drawing.max_angle_deg=5.
            position.return_value=np.array([0.,0.,.078])
            angle.return_value=6.
            self.assertFalse(interception_contact(env)[1])
            angle.return_value=4.
            self.assertTrue(interception_contact(env)[1])

    def test_reaching_current_node_starts_motion_without_leaving(self):
        parameters=resolve_board_motion(0,dict(mode="arrival_boards"))
        self.assertEqual(parameters["phases"],["execution"])
        gate=DepartureCrossings(parameters)
        args=dict(curve=0,start_index=2,end_index=4,direction="clockwise",start_xy=[0,0])
        gate.update(**args,tip_xy=[0,0],drawing=False,time_s=0.)
        gate.update(**args,tip_xy=[.031,0],drawing=True,time_s=1.)
        self.assertFalse(gate.events)
        gate.update(**args,tip_xy=[.025,0],drawing=True,time_s=2.)
        self.assertEqual(len(gate.events),1)
        self.assertEqual(gate.events[0]["started_s"],2.)
        self.assertAlmostEqual(gate.events[0]["arrival_distance_m"],.025)
        for time in np.linspace(2,12,101):
            gate.update(**args,tip_xy=[.025,0],drawing=True,time_s=time)
        self.assertEqual(len(gate.events),1)
        offsets,_=gate.offsets(3.)
        np.testing.assert_allclose(offsets,[0,.035,0,0])
        times=np.linspace(2,7.5,1101)
        values=np.array([gate.offsets(t)[0] for t in times])
        self.assertLessEqual(np.abs(np.diff(values,axis=0)).max()/.005,.052501)
        np.testing.assert_array_equal(values[0],[0,0,0,0])
        np.testing.assert_array_equal(values[-1],[0,0,0,0])

    def test_saved_full_stroke_arrival_intercepts_execution_curve(self):
        targets=np.array([[0.,.08*i] for i in range(9)])
        center=targets.mean(axis=0)
        for scale in (.9,1.):
            for yaw in (-10.,10.):
                for side in (-1,1):
                    layout=resolve_route_layout(0,dict(version=3,spacing_scale=scale,
                        yaw_deg=yaw,translation_xy_m=[.05,0.],board_motion=dict(mode="arrival_boards",stroke_m=.1)))
                    nodes=execution_points(targets,center,layout)
                    rests=nodes[[1,3,5,7]]
                    gate=DepartureCrossings(layout['board_motion'])
                    hits=[]
                    for i,u in enumerate(np.linspace(0,1,45)):
                        seconds=i*.05
                        point=execution_points(np.array([side*.4*u*(1-u),.16*u]),center,layout)
                        gate.update(curve=0,start_index=0,end_index=2,
                            direction="clockwise" if side<0 else "counterclockwise",
                            tip_xy=point,start_xy=nodes[0],drawing=True,time_s=seconds)
                        boards,_=gate.positions(rests,nodes[-1]-nodes[0],seconds,scale)
                        np.testing.assert_allclose(boards[1:],rests[1:],atol=1e-12)
                        delta=np.abs((point-boards[0])@rotation(layout))-[.03,.015]
                        if np.linalg.norm(np.maximum(delta,0))<.008:
                            hits.append(seconds)
                    self.assertEqual(gate.events[0]['started_s'],0.)
                    self.assertTrue(hits)
                    self.assertGreater(min(hits),.5)

    def test_arrival_configuration_roundtrip_and_invalid_fields(self):
        layout=resolve_route_layout(0,dict(version=3,board_motion=dict(mode="arrival_boards")))
        self.assertEqual(layout['board_motion']['stroke_m'],.07)
        row=normalize_episode("RouteStick",dict(seed=0,difficulty="hard",oracle_policy_step=100,
                                               execution_route_layout=layout),0)
        self.assertEqual(normalize_episode("RouteStick",row,0),row)
        for bad in (dict(arrival_radius_m=0),dict(arrival_radius_m=.04),
                    dict(arrival_radius_m=True),dict(departure_radius_m=.035)):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                resolve_board_motion(0,dict(mode="arrival_boards",**bad))


if __name__=="__main__":
    unittest.main()
