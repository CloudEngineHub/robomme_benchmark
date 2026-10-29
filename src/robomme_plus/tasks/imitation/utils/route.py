import torch

from robomme_plus.common.logging_utils import logger


def generate_dynamic_walk(
    indices,
    steps=50,
    start_idx=None,
    allow_backtracking=True,
    generator=None,
    walk_config=None,
):
    """
    Generate random walk trajectory (supports PyTorch Generator).

    Args:
        indices (list): Discrete position points (e.g., [0, 2, 4...])
        steps (int): Total steps
        start_idx (int): Start index, random if None
        allow_backtracking (bool): Whether to allow immediate backtracking
        generator (torch.Generator): PyTorch generator for controlling random seed
    """

    neighbor_order = [-1, 1]
    if walk_config is not None:
        if (
            walk_config.get("node_indices") != list(indices)
            or walk_config.get("start_selection") != "randint"
            or walk_config.get("neighbor_order") != [-1, 1]
            or walk_config.get("force_reverse_at_endpoint") is not True
        ):
            raise ValueError(
                "generate_dynamic_walk only supports the original linear adjacency and endpoint reversal rules"
            )
        neighbor_order = walk_config["neighbor_order"]

    # 1. Initialization
    if start_idx is None:
        # [Modification] Use torch to generate random start point
        # randint returns tensor, need .item() to convert to python int
        start_idx = torch.randint(0, len(indices), (1,), generator=generator).item()

    start_val = indices[start_idx]
    logger.debug(f"   -> Random start index: {start_idx} (Value: {start_val})")

    history_idxs = [start_idx]

    # 2. Generation loop
    for _ in range(steps):
        current_idx = history_idxs[-1]
        prev_idx = history_idxs[-2] if len(history_idxs) > 1 else None

        # Find all physically reachable neighbors
        neighbors = []
        for offset in neighbor_order:
            neighbor = current_idx + offset
            if 0 <= neighbor < len(indices):
                neighbors.append(neighbor)

        # --- Core logic: Backtracking filter ---
        candidates = []
        if allow_backtracking:
            candidates = neighbors
        else:
            # Backtracking not allowed
            if prev_idx is not None:
                filtered = [n for n in neighbors if n != prev_idx]
                candidates = (
                    filtered if filtered else neighbors
                )  # Must backtrack if no way forward
            else:
                candidates = neighbors

        # [Modification] Use torch to select randomly from candidates
        # Principle: Randomly generate an index from 0 to len(candidates)-1
        rand_choice_idx = torch.randint(
            0, len(candidates), (1,), generator=generator
        ).item()
        next_idx = candidates[rand_choice_idx]

        history_idxs.append(next_idx)

    # Map back to real values
    path_values = [indices[i] for i in history_idxs]

    return path_values
