# -*- coding: utf-8 -*-
"""
obs_convert.py
==============
将 DemonstrationWrapper.reset() / step() 返回的 obs / info 中的
torch.Tensor 字段转换为 np.ndarray，eef_state_list 从 list[float]
转换为 np.float64 ndarray。

转换是 **原地（in-place）** 的：直接修改传入的 dict 并返回同一对象。
调用方可忽略返回值（等价于 void），也可以链式使用。

支持的 obs 字段：
    front_rgb_list          list[Tensor] → list[ndarray uint8]
    wrist_rgb_list          list[Tensor] → list[ndarray uint8]
    front_depth_list        list[Tensor] → list[ndarray int16]
    wrist_depth_list        list[Tensor] → list[ndarray int16]
    end_effector_pose_raw   list[dict{pose,quat,rpy: Tensor}]
                              → list[dict{pose,quat,rpy: ndarray float32}]
    eef_state_list          list[list[float]] → list[ndarray float64]
    joint_state_list        不变（已是 ndarray）
    gripper_state_list      不变（已是 ndarray）
    front_camera_extrinsic_list  list[Tensor] → list[ndarray float32]
    wrist_camera_extrinsic_list  list[Tensor] → list[ndarray float32]
    maniskill_obs           不转换

支持的 info 字段：
    front_camera_intrinsic  Tensor → ndarray float32
    wrist_camera_intrinsic  Tensor → ndarray float32
    其余字段（str / None / list[str]）不变
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _tensor_to_numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    """将单个 Tensor 转换为指定 dtype 的 ndarray；若已是 ndarray 则只转换 dtype。"""
    if _HAS_TORCH and isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def _convert_list_of_tensors(lst: list, dtype: np.dtype) -> list:
    """将 list 中每个 Tensor 元素转换为指定 dtype 的 ndarray。"""
    return [_tensor_to_numpy(item, dtype) for item in lst]


def _convert_eef_pose_list(lst: list) -> list:
    """
    将 list[dict] 中每个 dict 的 pose / quat / rpy 键从 Tensor 转换为 float32 ndarray。
    其余键（如有）保持不变。
    """
    result = []
    for item in lst:
        if isinstance(item, dict):
            new_item = dict(item)
            for key in ("pose", "quat", "rpy"):
                if key in new_item:
                    new_item[key] = _tensor_to_numpy(new_item[key], np.float32)
            result.append(new_item)
        else:
            result.append(item)
    return result


def _convert_eef_state_list(lst: list) -> list:
    """
    将 list[list[float]] 转换为 list[ndarray float64]。
    若元素已经是 ndarray 则只做 dtype 转换。
    """
    return [np.asarray(item, dtype=np.float64) for item in lst]


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def convert_obs(obs: dict) -> dict:
    """
    原地转换 obs dict 中各字段。

    Parameters
    ----------
    obs : dict
        DemonstrationWrapper.reset() 或 step() 返回的 obs 字典。

    Returns
    -------
    dict
        与 obs 同一对象（原地修改后返回）。
    """
    # RGB：uint8
    if "front_rgb_list" in obs and obs["front_rgb_list"]:
        obs["front_rgb_list"] = _convert_list_of_tensors(obs["front_rgb_list"], np.uint8)
    if "wrist_rgb_list" in obs and obs["wrist_rgb_list"]:
        obs["wrist_rgb_list"] = _convert_list_of_tensors(obs["wrist_rgb_list"], np.uint8)

    # Depth：int16
    if "front_depth_list" in obs and obs["front_depth_list"]:
        obs["front_depth_list"] = _convert_list_of_tensors(obs["front_depth_list"], np.int16)
    if "wrist_depth_list" in obs and obs["wrist_depth_list"]:
        obs["wrist_depth_list"] = _convert_list_of_tensors(obs["wrist_depth_list"], np.int16)

    # end_effector_pose_raw：list[dict{pose,quat,rpy}] → float32
    if "end_effector_pose_raw" in obs and obs["end_effector_pose_raw"]:
        obs["end_effector_pose_raw"] = _convert_eef_pose_list(obs["end_effector_pose_raw"])

    # eef_state_list：list[list[float]] → list[ndarray float64]
    if "eef_state_list" in obs and obs["eef_state_list"]:
        obs["eef_state_list"] = _convert_eef_state_list(obs["eef_state_list"])

    # joint_state_list / gripper_state_list：不变（已是 ndarray）

    # camera extrinsics：float32
    if "front_camera_extrinsic_list" in obs and obs["front_camera_extrinsic_list"]:
        obs["front_camera_extrinsic_list"] = _convert_list_of_tensors(
            obs["front_camera_extrinsic_list"], np.float32
        )
    if "wrist_camera_extrinsic_list" in obs and obs["wrist_camera_extrinsic_list"]:
        obs["wrist_camera_extrinsic_list"] = _convert_list_of_tensors(
            obs["wrist_camera_extrinsic_list"], np.float32
        )

    # maniskill_obs 不转换
    return obs


def convert_info(info: dict) -> dict:
    """
    原地转换 info dict 中各相机内参字段（Tensor → ndarray float32）。
    其余字段（str / None / list[str]）保持不变。

    Parameters
    ----------
    info : dict
        DemonstrationWrapper.reset() 或 step() 返回的 info 字典。

    Returns
    -------
    dict
        与 info 同一对象（原地修改后返回）。
    """
    for key in ("front_camera_intrinsic", "wrist_camera_intrinsic"):
        if key in info:
            val = info[key]
            if _HAS_TORCH and isinstance(val, torch.Tensor):
                info[key] = _tensor_to_numpy(val, np.float32)
            elif isinstance(val, np.ndarray) and val.dtype != np.float32:
                info[key] = val.astype(np.float32, copy=False)
    return info
