import torch

from robomme_plus.common.logging_utils import logger


def grid_adjacency(R=3, C=3, diagonals=False, by_index=True):
    """
    R, C: rows and columns
    diagonals: False=4-connectivity; True=8-connectivity
    by_index: True returns with index (0..R*C-1) as key; False returns with coordinate (r,c) as key
    """
    # 4-connectivity directions
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # 8-connectivity adds four diagonal directions
    if diagonals:
        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    adj = {}
    for r in range(R):
        for c in range(C):
            nbrs = []
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C:
                    nbrs.append((nr, nc))

            key = r * C + c if by_index else (r, c)
            if by_index:
                adj[key] = [nr * C + nc for (nr, nc) in nbrs]
            else:
                adj[key] = nbrs
    return adj


def dfs_path(adj, start, target, generator=None, blocked_nodes=None):
    """
    Execute DFS to find path from start to target
    Returns visit_order, path and edges used during search

    Args:
        adj: adjacency list
        start: start node
        target: target node
        generator: torch.Generator for random neighbor selection
        blocked_nodes: list of node indices to avoid (cannot pass through these nodes)

    Returns:
        visit_order: list of all nodes visited in order
        path: final path from start to target
        edges_used: edges traversed during search
    """
    visited = set()
    visit_order = []  # Track all nodes visited in order
    path = []
    edges_used = []
    found = False

    # Convert blocked_nodes to set for O(1) lookup
    if blocked_nodes is None:
        blocked_nodes = set()
    else:
        blocked_nodes = set(blocked_nodes)

    def dfs_helper(node, current_path):
        nonlocal found
        if found:
            return

        visited.add(node)
        visit_order.append(node)  # Record visit order
        current_path.append(node)

        # Found target node
        if node == target:
            path.extend(current_path)
            found = True
            return

        # Get neighbors and optionally shuffle them
        neighbors = adj[node].copy()
        if generator is not None:
            # Use torch to randomly permute neighbors
            neighbors_tensor = torch.tensor(neighbors)
            perm = torch.randperm(len(neighbors), generator=generator)
            neighbors = neighbors_tensor[perm].tolist()

        # Continue searching
        for neighbor in neighbors:
            # Skip blocked nodes, already visited nodes
            if neighbor not in visited and neighbor not in blocked_nodes and not found:
                edges_used.append((node, neighbor))
                dfs_helper(neighbor, current_path)
                if found:
                    return
                current_path.pop()

    dfs_helper(start, [])
    return visit_order, path, edges_used


def find_path_0_to_8(
    start, target, R=3, C=3, diagonals=False, generator=None, blocked_nodes=None
):
    """Find path from start to target using DFS

    Args:
        start: start node
        target: target node
        R: number of rows
        C: number of columns
        diagonals: whether to use diagonal connections
        generator: torch.Generator for random neighbor selection
        blocked_nodes: list of node indices to avoid (cannot pass through these nodes)

    Returns:
        path: list of nodes in the final path from start to target (first return value)
        visit_order: list of all nodes visited in order during DFS
        edges_used: edges traversed during search
        adj: adjacency list
    """
    # Generate adjacency list
    adj = grid_adjacency(R, C, diagonals=diagonals, by_index=True)

    # Execute DFS to find path
    visit_order, path, edges_used = dfs_path(
        adj, start, target, generator=generator, blocked_nodes=blocked_nodes
    )

    if not path:
        logger.debug(f"❌ Cannot find path from {start} to {target}!")
        return None, None, None, None

    # Only print the path
    logger.debug(f"Path: {' → '.join(map(str, path))}")

    return path, visit_order, edges_used, adj
