"""Keep the goal cube and its native placement history at easy/medium."""

import copy


def single_cube_tasks(tasks, cube):
    """Keep the requested cube's complete pick/place pairs and the button event."""
    selected = []
    picked = None
    for entry in tasks:
        if not entry.get("demonstration"):
            selected.append(entry)
            continue
        name = entry["name"]
        if name == "pick up the cube":
            picked = entry["segment"]
            if picked is cube:
                selected.append(entry)
        elif name in (
            "drop the cube onto target",
            "put the cube back to its original position",
            "drop the cube onto table",
        ):
            if picked is None:
                raise ValueError("Demonstration placement has no preceding pickup")
            if picked is cube:
                selected.append(entry)
            picked = None
        else:
            selected.append(entry)
    return selected


def project_spec(source, cube_name, tasks):
    """Export the effective one-cube sequence while retaining its native answer."""
    document = copy.deepcopy(source)
    objects, actions = document["objects"], document["actions"]
    answer = objects["answer_demo_index"]
    objects["demo_ids"] = [objects["demo_ids"][answer]]
    objects["answer_demo_index"] = 0
    for key in ("demo_before_target_ids", "demo_after_target_ids"):
        if key in objects:
            objects[key] = [objects[key][answer]]
    if "visit_ids_by_object" in objects:
        visits = objects["visit_ids_by_object"][str(answer)]
        objects["visit_ids_by_object"] = {"0": visits}
        objects["visit_counts_by_object"] = [len(visits)]
        before = 0
        for index, entry in enumerate(tasks):
            if entry["name"] == "press the button":
                objects["button_after_visit_index"] = before
                actions["button_task_index"] = index
                break
            before += int(
                entry.get("demonstration")
                and entry["name"] == "drop the cube onto target"
            )
    if "place_sequence" in actions:
        actions["place_sequence"] = [
            [0, target]
            for owner, target in actions["place_sequence"]
            if owner == answer
        ]
        owners = iter(objects.get("extra_place_owner_ids", []))
        extras = {}
        for side in ("before", "after"):
            retained = []
            for _, target in sorted(
                objects.get("extra_place_target_ids", {}).get(side, {}).items(),
                key=lambda x: int(x[0]),
            ):
                if next(owners) == answer:
                    retained.append(target)
            if retained:
                extras[side] = {str(i): target for i, target in enumerate(retained)}
        objects["extra_place_target_ids"] = extras
        objects["extra_place_owner_ids"] = [0] * sum(
            len(items) for items in extras.values()
        )
    actions["target_placement_count"] = sum(
        e.get("demonstration") and e["name"] == "drop the cube onto target"
        for e in tasks
    )
    actions["return_pose_by_object_id"] = {
        cube_name: actions["return_pose_by_object_id"][cube_name]
    }
    return document


def select_one_cube(env, level):
    # _initialize_episode can run more than once; always project the same native
    # source, never project an already-renumbered answer a second time.
    if not hasattr(env, "_place_level_source_spec"):
        env._place_level_source_spec = env._spec.to_dict()
    source = env._place_level_source_spec
    env.task_list[:] = single_cube_tasks(env.task_list, env.target_cube)
    env.demo_cubes = [env.target_cube]
    effective = project_spec(source, env.target_cube.name, env.task_list)
    for section in ("objects", "actions"):
        env._spec.record(section, effective[section])
    env._spec.record(
        "custom_video_place_level",
        {
            "level": level,
            "projection": "requested cube only; other cube manipulation pairs removed",
            "source_demo_cube_count": len(source["objects"]["demo_ids"]),
            "source_answer_demo_index": source["objects"]["answer_demo_index"],
            "demo_cube_count": 1,
            "original_answer_identity_retained": True,
        },
    )
