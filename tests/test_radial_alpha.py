"""Test the OQ1 radial-scaling-exponent discriminates composition mechanisms."""
import numpy as np
from hypprobe.data.prontoqa_tree import _build_tree
from hypprobe.geometry.composition_test import composition_metrics


def _layout(kind, parent, depth, dim=96, seed=0):
    rng = np.random.default_rng(seed); pos = np.zeros((len(parent), dim))
    order = sorted(range(len(parent)), key=lambda i: depth[i])
    v = rng.standard_normal(dim); v /= np.linalg.norm(v)
    for i in order:
        p = parent[i]
        if p < 0:
            continue
        if kind == "aligned":
            pos[i] = pos[p] + v
        elif kind == "ortho":
            e = rng.standard_normal(dim); e /= np.linalg.norm(e); pos[i] = pos[p] + e
        elif kind == "cone":
            e = rng.standard_normal(dim); e /= np.linalg.norm(e); pos[i] = pos[p] + (0.5 ** depth[i]) * e
    return pos + 0.05 * rng.standard_normal(pos.shape)


def test_alpha_orders_mechanisms():
    parent, depth = _build_tree(2, 31)
    depth = np.array(depth)
    a = {}
    for kind in ("cone", "ortho", "aligned"):
        vals = []
        for s in range(5):
            m = composition_metrics(_layout(kind, parent, depth, seed=s), parent, depth)
            if m and isinstance(m["radial_alpha"], (int, float)):
                vals.append(m["radial_alpha"])
        a[kind] = np.mean(vals)
    # cone (radius saturates) < ortho (sqrt) < aligned (linear)
    assert a["cone"] < a["ortho"] < a["aligned"], a
    assert a["cone"] < 0.15 and a["aligned"] > 0.6, a
