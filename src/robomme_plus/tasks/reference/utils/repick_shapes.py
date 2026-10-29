"""Physical prisms with distinct overhead silhouettes for VideoRepick."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

SHAPES = ("cube", "moon", "triangle", "star", "concave", "convex")
ASSET_DIR = Path(__file__).with_name("assets") / "repick_shapes"


def shape_for_seed(seed):
    """One shared shape per episode, independent of count, color and target."""
    return SHAPES[int(np.random.default_rng(int(seed) + 41730).integers(len(SHAPES)))]


@lru_cache(maxsize=1)
def shape_assets():
    return json.loads((ASSET_DIR / "shapes.json").read_text())


def build_object(scene, *, shape, half_size, color, name, initial_pose):
    """Matching visual and compound-convex collision geometry, not a hidden cube.

    All meshes fit inside the native cube envelope. Concave cutouts remain
    empty: convex pieces cover the prism, rather than its convex hull.
    """
    import sapien
    from mani_skill.utils.building import actors

    if shape == "cube":
        actor = actors.build_cube(scene, half_size=half_size, color=color,
                                  name=name, initial_pose=initial_pose)
    else:
        assets = shape_assets()[shape]
        builder = scene.create_actor_builder()
        builder.initial_pose = initial_pose
        scale = [float(half_size)] * 3
        material = sapien.render.RenderMaterial(base_color=color, roughness=.65)
        builder.add_visual_from_file(str(ASSET_DIR / assets["mesh"]),
                                     scale=scale, material=material)
        for filename in assets["collision_parts"]:
            builder.add_convex_collision_from_file(str(ASSET_DIR / filename),
                                                   scale=scale)
        actor = builder.build_dynamic(name=name)
        body = actor._objs[0].find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
        if len(body.get_collision_shapes()) != len(assets["collision_parts"]):
            raise RuntimeError(f"Incomplete VideoRepick collision geometry for {shape}")
    actor._cube_half_size = float(half_size)
    actor._repick_shape = shape
    return actor
