"""Front-camera robot occlusion of the horizontal VideoPlace target disks."""

import numpy as np

from robomme_plus.common.sim import position

MAX_ARM_OCCLUSION = 0.5


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def camera_rays(shape, intrinsic, extrinsic):
    """World-space rays through pixel centers; extrinsic uses OpenCV axes."""
    rows, columns = np.indices(shape)
    pixels = np.stack((columns + .5, rows + .5, np.ones(shape)), axis=-1)
    intrinsic = np.asarray(intrinsic).reshape(3, 3)
    extrinsic = np.asarray(extrinsic).reshape(3, 4)
    rotation, translation = extrinsic[:, :3], extrinsic[:, 3]
    return -rotation.T @ translation, pixels @ np.linalg.inv(intrinsic).T @ rotation


def target_silhouette(origin, rays, center, radius, thickness=.005):
    """Rasterize the full target cylinder, including its edge, without occluders.

    VideoPlace rings remain horizontal. Ray/cylinder intersections account for
    perspective and target thickness without changing any scene or robot pose.
    """
    offset = origin - np.asarray(center)
    half = thickness / 2
    mask = np.zeros(rays.shape[:2], dtype=bool)
    for z in (-half, half):
        distance = np.divide(z - offset[2], rays[..., 2],
                             out=np.zeros(mask.shape), where=rays[..., 2] != 0)
        xy = offset[:2] + distance[..., None] * rays[..., :2]
        mask |= (distance > 0) & (np.square(xy).sum(axis=-1) <= radius ** 2)
    a = np.square(rays[..., :2]).sum(axis=-1)
    b = 2 * (rays[..., :2] * offset[:2]).sum(axis=-1)
    c = np.square(offset[:2]).sum() - radius ** 2
    discriminant = b * b - 4 * a * c
    root = np.sqrt(np.maximum(discriminant, 0))
    for sign in (-1, 1):
        distance = np.divide(-b + sign * root, 2 * a,
                             out=np.zeros(mask.shape), where=a > 0)
        height = offset[2] + distance * rays[..., 2]
        mask |= (discriminant >= 0) & (distance > 0) & (np.abs(height) <= half)
    return mask


def occlusion_counts(target_mask, robot_mask, *, maximum_fraction=MAX_ARM_OCCLUSION):
    total = int(np.count_nonzero(target_mask))
    blocked = int(np.count_nonzero(target_mask & robot_mask))
    return {
        "total_target_pixels": total,
        "arm_occluded_pixels": blocked,
        "arm_occluded_fraction": blocked / total if total else None,
        "allowed": total > 0 and blocked <= maximum_fraction * total,
    }


def front_camera_robot_mask(env):
    """Render one shared front-camera arm mask for an entire movement tick."""
    sensor = env._sensors["base_camera"]
    env.scene.update_render(update_human_render_cameras=False)
    sensor.capture()
    segmentation = array(sensor.get_obs(
        rgb=False, depth=False, position=False, segmentation=True
    )["segmentation"]).squeeze()
    if segmentation.ndim != 2:
        raise RuntimeError("Movement visibility requires one front camera")
    params = sensor.get_params()
    intrinsic = array(params["intrinsic_cv"]).reshape(3, 3)
    extrinsic = array(params["extrinsic_cv"]).reshape(3, 4)
    robot_ids = [int(array(link.per_scene_id).item()) for link in env.agent.robot.links]
    return np.isin(segmentation, robot_ids), intrinsic, extrinsic


class SwapVisibility:
    """Capture the current arm mask only when a new swap is being considered."""

    def __init__(self, env, radius):
        self.env, self.radius = env, radius
        self._camera_key = None

    def measure(self, actors):
        robot_mask, intrinsic, extrinsic = front_camera_robot_mask(self.env)
        key = (robot_mask.shape, intrinsic.tobytes(), extrinsic.tobytes())
        if key != self._camera_key:
            self._origin, self._rays = camera_rays(robot_mask.shape, intrinsic, extrinsic)
            self._camera_key = key
            self._masks = {}
        result = []
        for actor in actors:
            center = position(actor, initial=False)
            key = tuple(center.round(6))
            if key not in self._masks:
                self._masks[key] = target_silhouette(self._origin, self._rays, center, self.radius)
            result.append(occlusion_counts(self._masks[key], robot_mask))
        return result
