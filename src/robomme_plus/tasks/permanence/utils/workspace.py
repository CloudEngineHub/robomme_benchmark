"""Seeded motion modes and recorded movement bounds for ButtonUnmaskSwap."""

import copy
import hashlib
import numpy as np


COMPACT_WORKSPACE = dict(
    bounds_xy_m=[[-.40, .16], [-.29, .29]],
    central_bounds_xy_m=[[-.32, .16], [-.28, .28]],
    board_x_m=.20,
)


def resolve_motion_version(value=None):
    """New recordings defer pickup pauses for button presses; older versions replay."""
    if value is None:
        return 6
    if type(value) is not int or value not in (1, 2, 3, 4, 5, 6):
        raise ValueError('continuous_motion_version must be 1, 2, 3, 4, 5 or 6')
    return value


def resolve_continuous_motion(task, difficulty, seed, value=None):
    """Sample new episodes once; explicit booleans restore recorded behavior."""
    if value is not None and type(value) is not bool:
        raise ValueError('continuous_motion must be a boolean or None')
    if task != 'ButtonUnmaskSwap':
        if value:
            raise ValueError('continuous_motion is only supported for ButtonUnmaskSwap')
        return False
    if difficulty not in ('easy', 'medium', 'hard'):
        raise ValueError('Unknown ButtonUnmaskSwap difficulty')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if value is not None:
        return value
    if difficulty == 'easy':
        return False
    digest = hashlib.sha256(f'button-swap-continuous-v1/{difficulty}/{seed}'.encode()).digest()
    return digest[0] < 128


def resolve_workspace(value=None):
    if isinstance(value, str) and value == 'legacy':
        return None
    result = copy.deepcopy(COMPACT_WORKSPACE if value is None else value)
    if not isinstance(result, dict) or set(result) != set(COMPACT_WORKSPACE):
        raise ValueError('motion_workspace needs outer bounds, central bounds, and board_x_m')
    outer = np.asarray(result['bounds_xy_m'], dtype=float)
    central = np.asarray(result['central_bounds_xy_m'], dtype=float)
    if (outer.shape != (2, 2) or central.shape != (2, 2)
            or not np.isfinite(outer).all() or not np.isfinite(central).all()
            or np.any(np.diff(outer, axis=1) <= 0) or np.any(np.diff(central, axis=1) <= 0)
            or np.any(central[:, 0] < outer[:, 0]) or np.any(central[:, 1] > outer[:, 1])):
        raise ValueError('Central motion bounds must fit inside the outer workspace')
    board = result['board_x_m']
    if isinstance(board, bool) or not isinstance(board, (int, float)) or not np.isfinite(board):
        raise ValueError('board_x_m must be a finite number')
    return dict(bounds_xy_m=outer.tolist(), central_bounds_xy_m=central.tolist(), board_x_m=float(board))
