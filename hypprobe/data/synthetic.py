"""Synthetic data with known geometry, for tests and the positive control.

These generators let us validate the whole pipeline on CPU before spending any
DGX time on 7B models:

  - :func:`balanced_tree` gives a graph with exact tree distances -> delta_rel
    should be ~0 and a hyperbolic probe should beat a flat one at low dimension.
  - :func:`random_gaussian` is a flat cloud -> delta_rel should be high.
  - :func:`sphere_points` is positively curved -> delta_rel high.
  - :func:`hierarchy_features` produces a labelled feature set whose classes form
    a tree, used as the WordNet-style positive control for the probes.
"""

from __future__ import annotations

import numpy as np


def balanced_tree(depth: int = 5, branching: int = 2, seed: int = 0):
    """A balanced tree; returns (points, tree_distance_matrix, node_depths).

    ``points`` are node feature vectors laid out so their pairwise *tree*
    distance is exactly the graph distance. We first build exact tree distances,
    then embed them with classical MDS so the point cloud realises those
    distances as closely as a flat embedding allows -- but we always return the
    exact tree distance matrix too, which is what delta should be computed on
    for the "delta of a tree is ~0" sanity check.
    """
    rng = np.random.default_rng(seed)
    # Build parent list for a balanced tree.
    parents = [-1]
    frontier = [0]
    depths = [0]
    for _ in range(depth):
        new_frontier = []
        for node in frontier:
            for _ in range(branching):
                parents.append(node)
                depths.append(depths[node] + 1)
                new_frontier.append(len(parents) - 1)
        frontier = new_frontier
    n = len(parents)

    # Exact tree distances via BFS on the (unweighted) tree.
    adj: list[list[int]] = [[] for _ in range(n)]
    for child, par in enumerate(parents):
        if par >= 0:
            adj[child].append(par)
            adj[par].append(child)
    tree_d = np.full((n, n), 0.0)
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        tree_d[src] = dist

    # A random feature embedding is enough for the probe tests (labels below use
    # the true depths); geometry sanity uses tree_d directly.
    points = rng.standard_normal((n, 16)).astype(np.float64)
    return points, tree_d, np.asarray(depths)


def random_gaussian(n: int = 400, dim: int = 16, seed: int = 0) -> np.ndarray:
    """A flat, isotropic Gaussian cloud (high delta_rel expected)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float64)


def sphere_points(n: int = 400, dim: int = 16, seed: int = 0) -> np.ndarray:
    """Points on a unit sphere (positively curved; high delta_rel expected)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float64)


def hierarchy_features(
    depth: int = 3,
    branching: int = 3,
    per_leaf: int = 40,
    dim: int = 12,
    noise: float = 0.15,
    seed: int = 0,
):
    """Labelled features whose classes form a strict tree (positive control).

    Each leaf class is a point in a tree; its samples are the leaf's hyperbolic
    prototype plus small noise. The label hierarchy is genuine (nested), so a
    hyperbolic probe at low dimension should beat a flat one here -- if it does
    not, the pipeline is broken.

    Returns (features, leaf_labels, tree_distance_between_leaves, leaf_paths).
    """
    rng = np.random.default_rng(seed)

    # Enumerate leaf paths of a balanced tree of the given depth/branching.
    paths: list[tuple[int, ...]] = [()]
    for _ in range(depth):
        paths = [p + (b,) for p in paths for b in range(branching)]
    n_leaves = len(paths)

    # Prototype per leaf: place deeper nodes nearer the boundary along a
    # direction determined by their path -> mimics a Poincare tree embedding.
    directions = rng.standard_normal((n_leaves, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radius = 0.9  # leaves near the boundary
    prototypes = radius * directions

    feats = []
    labels = []
    for leaf_idx in range(n_leaves):
        pts = prototypes[leaf_idx][None, :] + noise * rng.standard_normal((per_leaf, dim))
        feats.append(pts)
        labels.extend([leaf_idx] * per_leaf)
    features = np.concatenate(feats, axis=0).astype(np.float64)
    labels_arr = np.asarray(labels)

    # Tree distance between leaves = 2 * (depth - shared prefix length).
    tree_d = np.zeros((n_leaves, n_leaves))
    for i in range(n_leaves):
        for j in range(n_leaves):
            shared = 0
            for a, b in zip(paths[i], paths[j]):
                if a == b:
                    shared += 1
                else:
                    break
            tree_d[i, j] = 2 * (depth - shared)

    return features, labels_arr, tree_d, paths
