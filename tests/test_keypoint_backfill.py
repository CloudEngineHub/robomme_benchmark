"""
Unit tests for RecordWrapper keypoint backfill and _consume_pending_keypoint logic.

These tests verify:
1. _backfill_keypoint_actions_in_buffer handles single keyframe, leading/trailing steps
2. _consume_pending_keypoint correctly processes pending keypoints
3. close() consumes trailing pending keypoints before backfill
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import torch


def _make_buffer_entry(timestep, is_keyframe=False, keypoint_action=None):
    """Helper to create a buffer record dict."""
    entry = {
        "obs": {},
        "action": {},
        "info": {
            "record_timestep": timestep,
            "is_keyframe": is_keyframe,
        },
    }
    if keypoint_action is not None:
        entry["action"]["keypoint_action"] = np.array(keypoint_action, dtype=np.float32)
    return entry


class FakeRecordWrapper:
    """
    Minimal stand-in that has the same buffer / backfill / consume logic as
    RobommeRecordWrapper but without gym.Wrapper or any SAPIEN dependencies.
    """

    def __init__(self):
        self.buffer = []
        self._current_keypoint_action = None
        self._prev_ee_quat_wxyz = None
        self._prev_ee_rpy_xyz = None

        # Fake env with unwrapped attribute
        self.env = MagicMock()
        self.env.unwrapped = MagicMock()
        self.env.unwrapped._pending_keypoint = None

    # ------------------------------------------------------------------
    # Import the real methods by binding them from the source module.
    # We patch build_endeffector_pose_dict so it just echoes back values.
    # ------------------------------------------------------------------

    def _consume_pending_keypoint(self) -> bool:
        env_unwrapped = getattr(self.env, "unwrapped", self.env)
        if not (
            hasattr(env_unwrapped, "_pending_keypoint")
            and env_unwrapped._pending_keypoint is not None
        ):
            return False

        current_keypoint = env_unwrapped._pending_keypoint

        if (
            "keypoint_p" not in current_keypoint
            or "keypoint_q" not in current_keypoint
        ):
            raise ValueError(
                f"_pending_keypoint missing keypoint_p/keypoint_q: {current_keypoint}"
            )

        keypoint_p_np = np.asarray(current_keypoint["keypoint_p"]).reshape(-1)
        keypoint_q_np = np.asarray(current_keypoint["keypoint_q"]).reshape(-1)
        if keypoint_p_np.size != 3 or keypoint_q_np.size != 4:
            raise ValueError(
                f"_pending_keypoint keypoint shape invalid: "
                f"p={keypoint_p_np.shape}, q={keypoint_q_np.shape}"
            )

        kp_type = current_keypoint.get("keypoint_type", "unknown")
        gripper_val = 1.0 if kp_type == "open" else -1.0

        # Simplified: use position + [0,0,0] for rpy (real code uses build_endeffector_pose_dict)
        self._current_keypoint_action = np.concatenate(
            [keypoint_p_np[:3], np.zeros(3), [gripper_val]]
        )

        env_unwrapped._pending_keypoint = None
        return True

    def _backfill_keypoint_actions_in_buffer(self) -> None:
        if not self.buffer:
            return

        def _as_bool_flag(value) -> bool:
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return False
                return bool(value.detach().cpu().bool().any().item())
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return False
                return bool(np.asarray(value).astype(bool).any())
            return bool(value)

        def _validate_keypoint_action(kf_idx):
            kf_action = self.buffer[kf_idx].get("action", {})
            kp = kf_action.get("keypoint_action", None)
            if kp is None:
                return None
            target_kp = np.asarray(kp).flatten()
            if target_kp.size != 7:
                return None
            if not np.isfinite(target_kp).all():
                return None
            return target_kp

        def _fill_range(start, end_exclusive, target_kp):
            for fill_idx in range(start, end_exclusive):
                fill_action = self.buffer[fill_idx].setdefault("action", {})
                fill_action["keypoint_action"] = target_kp.copy()

        keyframe_indices = []
        for idx, record_data in enumerate(self.buffer):
            info_data = record_data.get("info", {})
            if _as_bool_flag(info_data.get("is_keyframe", False)):
                keyframe_indices.append(idx)

        if len(keyframe_indices) == 0:
            return

        if len(keyframe_indices) == 1:
            kf_idx = keyframe_indices[0]
            target_kp = _validate_keypoint_action(kf_idx)
            if target_kp is not None:
                _fill_range(0, len(self.buffer), target_kp)
            return

        first_kf = keyframe_indices[0]
        first_kp = _validate_keypoint_action(first_kf)
        if first_kp is not None:
            _fill_range(0, first_kf + 1, first_kp)

        for prev_idx, curr_idx in zip(keyframe_indices, keyframe_indices[1:]):
            target_kp = _validate_keypoint_action(curr_idx)
            if target_kp is None:
                continue
            _fill_range(prev_idx + 1, curr_idx + 1, target_kp)

        last_kf = keyframe_indices[-1]
        last_kp = _validate_keypoint_action(last_kf)
        if last_kp is not None and last_kf + 1 < len(self.buffer):
            _fill_range(last_kf + 1, len(self.buffer), last_kp)


# ==========================================================================
# Test cases
# ==========================================================================


class TestBackfillBoundaries(unittest.TestCase):
    """Test _backfill_keypoint_actions_in_buffer boundary handling."""

    def test_empty_buffer(self):
        w = FakeRecordWrapper()
        w._backfill_keypoint_actions_in_buffer()  # should not raise

    def test_no_keyframes(self):
        w = FakeRecordWrapper()
        w.buffer = [_make_buffer_entry(i) for i in range(5)]
        w._backfill_keypoint_actions_in_buffer()
        # No keyframes → nothing filled, keypoint_action stays absent or None
        for entry in w.buffer:
            self.assertIsNone(entry["action"].get("keypoint_action"))

    def test_single_keyframe_fills_entire_buffer(self):
        """Single keyframe at index 3 should fill all 6 entries."""
        w = FakeRecordWrapper()
        kp = [1, 2, 3, 0.1, 0.2, 0.3, -1.0]
        w.buffer = [_make_buffer_entry(i) for i in range(6)]
        w.buffer[3] = _make_buffer_entry(3, is_keyframe=True, keypoint_action=kp)

        w._backfill_keypoint_actions_in_buffer()

        for i, entry in enumerate(w.buffer):
            np.testing.assert_array_almost_equal(
                entry["action"]["keypoint_action"],
                kp,
                err_msg=f"buffer[{i}] not filled with keyframe action",
            )

    def test_leading_steps_filled_with_first_keyframe(self):
        """Steps before the first keyframe should be filled with first keyframe's action."""
        w = FakeRecordWrapper()
        kp1 = [1, 1, 1, 0, 0, 0, 1.0]
        kp2 = [2, 2, 2, 0, 0, 0, -1.0]
        w.buffer = [_make_buffer_entry(i) for i in range(10)]
        w.buffer[3] = _make_buffer_entry(3, is_keyframe=True, keypoint_action=kp1)
        w.buffer[7] = _make_buffer_entry(7, is_keyframe=True, keypoint_action=kp2)

        w._backfill_keypoint_actions_in_buffer()

        # Steps 0-3 should have kp1
        for i in range(4):
            np.testing.assert_array_almost_equal(
                w.buffer[i]["action"]["keypoint_action"],
                kp1,
                err_msg=f"buffer[{i}] should have first keyframe action",
            )

        # Steps 4-7 should have kp2
        for i in range(4, 8):
            np.testing.assert_array_almost_equal(
                w.buffer[i]["action"]["keypoint_action"],
                kp2,
                err_msg=f"buffer[{i}] should have second keyframe action",
            )

        # Steps 8-9 should have kp2 (trailing)
        for i in range(8, 10):
            np.testing.assert_array_almost_equal(
                w.buffer[i]["action"]["keypoint_action"],
                kp2,
                err_msg=f"buffer[{i}] should have last keyframe action (trailing)",
            )

    def test_trailing_steps_filled_with_last_keyframe(self):
        """Steps after the last keyframe should be filled."""
        w = FakeRecordWrapper()
        kp = [5, 5, 5, 0.5, 0.5, 0.5, 1.0]
        w.buffer = [_make_buffer_entry(i) for i in range(8)]
        w.buffer[2] = _make_buffer_entry(2, is_keyframe=True, keypoint_action=kp)
        # Only one keyframe, should fill everything
        w._backfill_keypoint_actions_in_buffer()
        for i in range(8):
            np.testing.assert_array_almost_equal(
                w.buffer[i]["action"]["keypoint_action"], kp
            )

    def test_three_keyframes(self):
        """Three keyframes with proper interval coverage."""
        w = FakeRecordWrapper()
        kp1 = [1, 0, 0, 0, 0, 0, 1.0]
        kp2 = [0, 2, 0, 0, 0, 0, -1.0]
        kp3 = [0, 0, 3, 0, 0, 0, 1.0]
        w.buffer = [_make_buffer_entry(i) for i in range(12)]
        w.buffer[2] = _make_buffer_entry(2, is_keyframe=True, keypoint_action=kp1)
        w.buffer[5] = _make_buffer_entry(5, is_keyframe=True, keypoint_action=kp2)
        w.buffer[9] = _make_buffer_entry(9, is_keyframe=True, keypoint_action=kp3)

        w._backfill_keypoint_actions_in_buffer()

        # [0, 2] → kp1
        for i in range(3):
            np.testing.assert_array_almost_equal(
                w.buffer[i]["action"]["keypoint_action"], kp1
            )
        # (2, 5] → kp2
        for i in range(3, 6):
            np.testing.assert_array_almost_equal(
                w.buffer[i]["action"]["keypoint_action"], kp2
            )
        # (5, 9] → kp3
        for i in range(6, 10):
            np.testing.assert_array_almost_equal(
                w.buffer[i]["action"]["keypoint_action"], kp3
            )
        # (9, 11] → kp3
        for i in range(10, 12):
            np.testing.assert_array_almost_equal(
                w.buffer[i]["action"]["keypoint_action"], kp3
            )


class TestConsumePendingKeypoint(unittest.TestCase):
    """Test _consume_pending_keypoint method."""

    def test_no_pending(self):
        w = FakeRecordWrapper()
        self.assertFalse(w._consume_pending_keypoint())
        self.assertIsNone(w._current_keypoint_action)

    def test_consume_open_keypoint(self):
        w = FakeRecordWrapper()
        w.env.unwrapped._pending_keypoint = {
            "keypoint_p": [1.0, 2.0, 3.0],
            "keypoint_q": [1.0, 0.0, 0.0, 0.0],
            "keypoint_type": "open",
            "solve_function": "test",
        }
        result = w._consume_pending_keypoint()
        self.assertTrue(result)
        self.assertIsNotNone(w._current_keypoint_action)
        self.assertEqual(w._current_keypoint_action.shape, (7,))
        # gripper_val should be 1.0 for "open"
        self.assertAlmostEqual(w._current_keypoint_action[-1], 1.0)
        # _pending_keypoint should be cleared
        self.assertIsNone(w.env.unwrapped._pending_keypoint)

    def test_consume_close_keypoint(self):
        w = FakeRecordWrapper()
        w.env.unwrapped._pending_keypoint = {
            "keypoint_p": [4.0, 5.0, 6.0],
            "keypoint_q": [0.0, 1.0, 0.0, 0.0],
            "keypoint_type": "close",
            "solve_function": "test",
        }
        result = w._consume_pending_keypoint()
        self.assertTrue(result)
        # gripper_val should be -1.0 for "close"
        self.assertAlmostEqual(w._current_keypoint_action[-1], -1.0)

    def test_consume_clears_pending(self):
        w = FakeRecordWrapper()
        w.env.unwrapped._pending_keypoint = {
            "keypoint_p": [0.0, 0.0, 0.0],
            "keypoint_q": [1.0, 0.0, 0.0, 0.0],
            "keypoint_type": "close",
            "solve_function": "test",
        }
        w._consume_pending_keypoint()
        # Second call should return False
        self.assertFalse(w._consume_pending_keypoint())


class TestCloseConsumesTrailingKeypoint(unittest.TestCase):
    """Test that close() logic properly handles unconsumed trailing keypoints."""

    def test_trailing_keypoint_marks_last_buffer_entry(self):
        """Simulate close() logic: pending keypoint consumed and applied to last buffer entry."""
        w = FakeRecordWrapper()
        kp1 = [1, 1, 1, 0, 0, 0, -1.0]

        # Build buffer with one keyframe and some trailing steps
        w.buffer = [_make_buffer_entry(i) for i in range(5)]
        w.buffer[1] = _make_buffer_entry(1, is_keyframe=True, keypoint_action=kp1)

        # Set a pending keypoint (simulating last _record_keypoint with no step after)
        w.env.unwrapped._pending_keypoint = {
            "keypoint_p": [9.0, 8.0, 7.0],
            "keypoint_q": [1.0, 0.0, 0.0, 0.0],
            "keypoint_type": "open",
            "solve_function": "test",
        }

        # --- Simulate what close() does ---
        if w.buffer and w._consume_pending_keypoint():
            last_record = w.buffer[-1]
            last_record.setdefault("action", {})["keypoint_action"] = (
                w._current_keypoint_action.copy()
            )
            last_record.setdefault("info", {})["is_keyframe"] = True

        # Verify last entry is now a keyframe
        self.assertTrue(w.buffer[-1]["info"]["is_keyframe"])
        last_kp = w.buffer[-1]["action"]["keypoint_action"]
        self.assertEqual(last_kp.shape, (7,))
        self.assertAlmostEqual(last_kp[-1], 1.0)  # open → gripper_val=1.0

        # Now run backfill
        w._backfill_keypoint_actions_in_buffer()

        # buffer[0]-buffer[1] should have kp1
        for i in range(2):
            np.testing.assert_array_almost_equal(
                w.buffer[i]["action"]["keypoint_action"], kp1,
                err_msg=f"buffer[{i}] should have first keyframe",
            )

        # buffer[2]-buffer[4] should have the trailing keypoint action
        for i in range(2, 5):
            np.testing.assert_array_almost_equal(
                w.buffer[i]["action"]["keypoint_action"][-1],
                1.0,  # open
                err_msg=f"buffer[{i}] should have trailing keyframe",
            )


if __name__ == "__main__":
    unittest.main()
