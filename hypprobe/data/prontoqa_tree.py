"""Branching-ontology ProntoQA with a RETAINED ground-truth tree.

Why this exists (see PREREGISTER3.md §0): the original ``build_prontoqa``
generates ``chain = rng.sample(stems, depth+1)`` — a **linear chain**. A path is
a degenerate tree with branching factor 1, and a path embeds *isometrically*
into 1-D Euclidean space, so hyperbolic geometry can buy nothing on it. A null
hyperbolic advantage on linear-chain ProntoQA is therefore forced by the
stimulus, not discovered in the model. This generator fixes that by making
**branching factor the independent variable** and by keeping the ground-truth
is-a tree in every record (the old generator threw it away).

Each prompt is a set of ``Every <child> is a <parent>.`` premises (shuffled, so
the structure is a genuine graph-reasoning task and not a left-to-right chain),
a placement fact ``<Entity> is a <leaf>.``, and a True/False query about an
ancestor (True) or a non-ancestor (False). We store ``tree_meta`` with the full
node/parent/depth structure so downstream code can recover exact pairwise tree
distances between the *concepts inside one prompt* — the target the tree probe
decodes.

Conditions produced by :func:`build_prontoqa_tree_all` (one combined dataset):
  * fictional (nonce) names at branching 1, 2, 3 — the branching sweep, with
    **matched node count** (a tree with N nodes always has N-1 edges, so premise
    count and token count are matched across branching automatically).
  * real (curated is-a) names — the "stored taxonomy" arm for the
    in-context-vs-retrieved contrast; the is-a claims are true in the world so
    the model *can* retrieve them.
"""

from __future__ import annotations

import random

import numpy as np

# --- fictional predicate pool (PrOntoQA-style nonce words) --------------------
# Enough distinct words that a ~15-node tree can be labelled without repeats.
NONCE = [
    "yumpus", "wumpus", "jompus", "zumpus", "numpus", "vumpus", "tumpus",
    "rompus", "dumpus", "sterpus", "lorpus", "grimpus", "shumpus", "brimpus",
    "gorpus", "lempus", "twmpus", "frompus", "daumpus", "kirpus", "melpus",
    "boompus", "charpus", "delpus", "empus", "florpus", "gimpus", "harpus",
    "irpus", "jelpus", "klorpus", "lumpus", "morpus", "norpus", "oompus",
    "plimpus", "qucompus", "relpus", "sompus", "torpus", "urpus", "vorpus",
    "welpus", "xompus", "yorpus", "zelpus", "arpus", "blompus",
]

ENTITIES = ["Max", "Alex", "Sam", "Polly", "Rex", "Fae", "Wren", "Stella",
            "Jo", "Kai", "Nia", "Uma"]

# --- curated REAL is-a taxonomy (no downloads; true-in-world hypernymy) --------
# Nested dict = is-a tree. Leaves are [] (concrete kinds). Deep + broad enough to
# sample connected ~15-node subtrees with real branching. The claim
# "Every dog is a mammal" etc. is true, so a model may *retrieve* it.
_REAL_TAXONOMY = {
    "animal": {
        "mammal": {
            "carnivore": {"dog": {}, "cat": {}, "wolf": {}, "bear": {}},
            "primate": {"human": {}, "monkey": {}, "gorilla": {}},
            "rodent": {"mouse": {}, "rat": {}, "squirrel": {}},
            "ungulate": {"horse": {}, "cow": {}, "deer": {}},
        },
        "bird": {
            "raptor": {"eagle": {}, "hawk": {}, "owl": {}},
            "songbird": {"sparrow": {}, "robin": {}, "finch": {}},
            "waterfowl": {"duck": {}, "goose": {}, "swan": {}},
        },
        "fish": {
            "shark": {}, "salmon": {}, "tuna": {}, "trout": {},
        },
        "reptile": {"snake": {}, "lizard": {}, "turtle": {}, "crocodile": {}},
    },
    "plant": {
        "tree": {"oak": {}, "pine": {}, "maple": {}, "birch": {}},
        "flower": {"rose": {}, "tulip": {}, "daisy": {}, "orchid": {}},
        "grass": {"wheat": {}, "bamboo": {}, "corn": {}},
    },
    "vehicle": {
        "car": {"sedan": {}, "coupe": {}, "hatchback": {}},
        "truck": {"pickup": {}, "van": {}, "lorry": {}},
        "aircraft": {"airplane": {}, "helicopter": {}, "glider": {}},
        "vessel": {"boat": {}, "ship": {}, "canoe": {}},
    },
}


def _build_tree(branching: int, n_nodes: int):
    """BFS-grow a (nearly) balanced tree with EXACTLY ``n_nodes`` nodes.

    Returns (parent, depth). ``branching=1`` yields a linear chain of ``n_nodes``
    (the negative-control path). Node 0 is the root.
    """
    parent = [-1]
    depth = [0]
    frontier = [0]
    while len(parent) < n_nodes:
        nxt = []
        for node in frontier:
            for _ in range(branching):
                if len(parent) >= n_nodes:
                    break
                parent.append(node)
                depth.append(depth[node] + 1)
                nxt.append(len(parent) - 1)
            if len(parent) >= n_nodes:
                break
        frontier = nxt or frontier
    return parent, depth


def _tree_distance(parent: list[int]) -> np.ndarray:
    """Exact unweighted path-length distance matrix over tree nodes (N, N)."""
    n = len(parent)
    adj: list[list[int]] = [[] for _ in range(n)]
    for child, par in enumerate(parent):
        if par >= 0:
            adj[child].append(par)
            adj[par].append(child)
    D = np.zeros((n, n))
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
        D[src] = dist
    return D


def _ancestors(node: int, parent: list[int]) -> set[int]:
    """Proper ancestors of ``node`` (excludes the node itself)."""
    out = set()
    p = parent[node]
    while p >= 0:
        out.add(p)
        p = parent[p]
    return out


def _leaves(parent: list[int]) -> list[int]:
    n = len(parent)
    has_child = [False] * n
    for c, p in enumerate(parent):
        if p >= 0:
            has_child[p] = True
    return [i for i in range(n) if not has_child[i]]


def _flatten_taxonomy(tax: dict, parent_name: str | None = None,
                      names=None, parent=None, name_to_id=None):
    """Flatten the nested real taxonomy into (names, parent) with node ids."""
    if names is None:
        names, parent, name_to_id = [], [], {}
    for key, sub in tax.items():
        nid = len(names)
        names.append(key)
        name_to_id[key] = nid
        parent.append(name_to_id[parent_name] if parent_name is not None else -1)
        _flatten_taxonomy(sub, key, names, parent, name_to_id)
    return names, parent, name_to_id


def _real_subtree(rng: random.Random, n_nodes: int):
    """Sample a connected ~n_nodes subtree of the curated real taxonomy.

    Pick an internal node with enough descendants as the subtree root, take a
    BFS-truncated descendant set, and relabel to a compact 0..k-1 tree. Returns
    (names, parent, depth) with real English names in true is-a relations.
    """
    full_names, full_parent, _ = _flatten_taxonomy(_REAL_TAXONOMY)
    n = len(full_names)
    children: list[list[int]] = [[] for _ in range(n)]
    for c, p in enumerate(full_parent):
        if p >= 0:
            children[p].append(c)

    # descendant count per node
    def _desc_count(node):
        stack, cnt = [node], 0
        while stack:
            u = stack.pop()
            for v in children[u]:
                cnt += 1
                stack.append(v)
        return cnt

    candidates = [i for i in range(n) if _desc_count(i) + 1 >= min(n_nodes, 6)]
    root = rng.choice(candidates) if candidates else 0

    # BFS from root, truncating at n_nodes
    order = [root]
    queue = [root]
    head = 0
    while head < len(queue) and len(order) < n_nodes:
        u = queue[head]
        head += 1
        for v in children[u]:
            if len(order) >= n_nodes:
                break
            order.append(v)
            queue.append(v)
    order_set = set(order)
    remap = {old: new for new, old in enumerate(order)}
    names = [full_names[o] for o in order]
    parent = []
    for o in order:
        p = full_parent[o]
        parent.append(remap[p] if p in order_set else -1)
    # recompute depth from the remapped parents
    depth = [0] * len(order)
    for i in range(len(order)):
        d, p = 0, parent[i]
        while p >= 0:
            d += 1
            p = parent[p]
        depth[i] = d
    return names, parent, depth


def _make_one(rng: random.Random, branching: int, naming: str, n_nodes: int,
              idx: int) -> dict:
    """Build a single branching-ontology prompt with retained tree_meta."""
    if naming == "real":
        names, parent, depth = _real_subtree(rng, n_nodes)
    else:
        parent, depth = _build_tree(branching, n_nodes)
        pool = rng.sample(NONCE, len(parent))
        names = list(pool)
    n = len(parent)

    # premises: child is-a parent, SHUFFLED (graph reasoning, not a chain).
    premises = [f"Every {names[c]} is a {names[parent[c]]}."
                for c in range(n) if parent[c] >= 0]
    rng.shuffle(premises)

    # place the entity at a leaf; query an ancestor (True) or non-ancestor (False)
    leaves = _leaves(parent)
    ent_node = rng.choice(leaves)
    ent = rng.choice(ENTITIES)
    anc = _ancestors(ent_node, parent)
    if idx % 2 == 0 and anc:
        target = rng.choice(sorted(anc))
        answer = 1
    else:
        non_anc = [i for i in range(n) if i != ent_node and i not in anc]
        target = rng.choice(non_anc) if non_anc else rng.choice(sorted(anc))
        answer = 0 if (non_anc) else 1

    fact = f"{ent} is a {names[ent_node]}."
    prompt = (" ".join(premises) + " " + fact + "\n"
              f"Question: Is it true or false that {ent} is a {names[target]}? "
              f"Reason step by step, then answer True or False.")

    tree_meta = {
        "branching": branching,
        "naming": naming,
        "n_nodes": n,
        "names": names,
        "parent": parent,
        "depth": depth,
        "entity": ent,
        "entity_node": ent_node,
        "target_node": target,
        "answer": answer,
    }
    naming_code = 1 if naming == "real" else 0
    return {
        "sample_id": f"tree_{naming}_b{branching}_{idx}",
        "prompt": prompt,
        "label": branching,                       # arm bookkeeping (not the decode target)
        "label_path": [naming_code, branching],
        "answer": answer,
        "tree_meta": tree_meta,
    }


def build_prontoqa_tree(branching: int = 2, naming: str = "fictional",
                        n_nodes: int = 15, n_prompts: int = 60,
                        seed: int = 0) -> list[dict]:
    """One condition: ``n_prompts`` branching-ontology prompts, tree retained."""
    rng = random.Random(seed * 1000 + branching * 10 + (1 if naming == "real" else 0))
    return [_make_one(rng, branching, naming, n_nodes, i) for i in range(n_prompts)]


def build_prontoqa_tree_all(n_prompts: int = 60, n_nodes: int = 15,
                            seed: int = 0) -> list[dict]:
    """All PREREGISTER3 conditions in one dataset (registered under this name).

    Fictional branching sweep b∈{1,2,3} (b=1 = negative-control path) at matched
    node count, plus a real (curated) arm for the in-context-vs-retrieved
    contrast. Downstream (:mod:`hypprobe.geometry.tree_probe`) separates arms by
    ``tree_meta.branching`` / ``tree_meta.naming``.
    """
    rows: list[dict] = []
    for b in (1, 2, 3):
        rows += build_prontoqa_tree(branching=b, naming="fictional",
                                    n_nodes=n_nodes, n_prompts=n_prompts, seed=seed)
    rows += build_prontoqa_tree(branching=2, naming="real",
                                n_nodes=n_nodes, n_prompts=n_prompts, seed=seed)
    # Interleave arms deterministically so a downstream --limit below the full
    # size still samples every arm (else the first arm would monopolise a small
    # limit). Arms are re-separated downstream via tree_meta, so order is free.
    random.Random(seed).shuffle(rows)
    return rows
