"""Front-camera arm occlusion of moving, oriented PickHighlight cubes."""

from itertools import product

import numpy as np

from .place_visibility import camera_rays, front_camera_robot_mask, occlusion_counts

MAX_ARM_OCCLUSION = .30


def rotation_matrix(quaternion):
    w, x, y, z = np.asarray(quaternion) / np.linalg.norm(quaternion)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def cube_silhouette(origin, rays, center, quaternion, half_size):
    """Ray/box intersection includes the cube's top and all visible side faces."""
    rotation = rotation_matrix(quaternion)
    local_origin = (origin - np.asarray(center)) @ rotation
    local_rays = rays @ rotation
    parallel = np.abs(local_rays) < 1e-12
    low = np.divide(-half_size-local_origin, local_rays,
                    out=np.full_like(local_rays, -np.inf), where=~parallel)
    high = np.divide(half_size-local_origin, local_rays,
                     out=np.full_like(local_rays, np.inf), where=~parallel)
    near = np.minimum(low, high).max(axis=-1)
    far = np.maximum(low, high).min(axis=-1)
    missed_parallel = (parallel & (np.abs(local_origin) > half_size)).any(axis=-1)
    return ~missed_parallel & (far > 0) & (far >= np.maximum(near, 0))


class CubeVisibility:
    def __init__(self, env, half_size):
        self.env, self.half_size = env, half_size
        self._camera_key = None
        self._corners = np.array(list(product((-half_size, half_size), repeat=3)))

    def capture(self):
        self.robot_mask, self.intrinsic, self.extrinsic = front_camera_robot_mask(self.env)
        key = (self.robot_mask.shape, self.intrinsic.tobytes(), self.extrinsic.tobytes())
        if key != self._camera_key:
            self.origin, self.rays = camera_rays(self.robot_mask.shape, self.intrinsic, self.extrinsic)
            self._camera_key = key

    def measure(self, center, quaternion):
        # Project a tight crop first: full-image box intersections per cube would
        # waste time when checking several groups on every control step.
        corners = self._corners @ rotation_matrix(quaternion).T + center
        camera = corners @ self.extrinsic[:, :3].T + self.extrinsic[:, 3]
        if (camera[:, 2] <= 0).any():
            return occlusion_counts(np.zeros((0, 0), bool), np.zeros((0, 0), bool),
                                    maximum_fraction=MAX_ARM_OCCLUSION)
        pixels = camera @ self.intrinsic.T
        pixels = pixels[:, :2] / pixels[:, 2:]
        height, width = self.robot_mask.shape
        x0, y0 = np.maximum(np.floor(pixels.min(axis=0)).astype(int)-1, 0)
        x1, y1 = np.minimum(np.ceil(pixels.max(axis=0)).astype(int)+1, [width, height])
        x0, y0 = min(x0, width), min(y0, height)
        x1, y1 = max(x1, 0), max(y1, 0)
        mask = cube_silhouette(self.origin, self.rays[y0:y1, x0:x1],
                               center, quaternion, self.half_size)
        return occlusion_counts(mask, self.robot_mask[y0:y1, x0:x1],
                                maximum_fraction=MAX_ARM_OCCLUSION)
