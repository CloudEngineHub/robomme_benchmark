"""A lost grasp-contact reading must not collect a cube from closed fingers."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from robomme_plus.tasks.counting.utils.controllers import ConveyorCounting
from robomme_plus.tasks.counting.utils.motion import FiniteSupply


class BinRelease(unittest.TestCase):
    def test_closed_fingers_do_not_collect_but_a_real_release_counts_once(self):
        def actor(name, point):
            return SimpleNamespace(name=name, pose=SimpleNamespace(p=np.array([point], float)))

        cube = actor('red_cube', [0, 0, .06])
        opening = False
        robot = SimpleNamespace(get_qpos=lambda: torch.full((1, 9), .04 if opening else .015),
                                get_drive_targets=lambda: torch.full((1, 9), .04 if opening else 0.))
        controller = ConveyorCounting.__new__(ConveyorCounting)
        controller.env = SimpleNamespace(board_with_hole=actor('bin', [0, 0, 0]),
            agent=SimpleNamespace(robot=robot, tcp=actor('tcp', [0, 0, .065]), is_grasping=lambda c: False))
        controller.items = [dict(actor=cube, status='held', color='red', ever_grasped=True)]
        controller.requested = {'red': 1}
        controller.ledger = FiniteSupply({'red': 1}, {'red': 1})
        controller.success, controller.failure = False, None
        controller.flags, controller.stop_with_button, controller.event = Mock(), Mock(), Mock()
        with patch('robomme_plus.tasks.counting.utils.controllers.park') as park:
            controller.evaluate()
            self.assertEqual(controller.ledger.collected, {})
            park.assert_not_called()
            opening = True
            controller.evaluate()
            controller.evaluate()
            self.assertEqual(controller.ledger.collected, {'red': 1})
            park.assert_called_once_with(cube, 0)
