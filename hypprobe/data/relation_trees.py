"""Relation-type generality: is the hierarchy signal is-a specific, or does any
structured ordering embed hyperbolically? Plus a FLAT negative control.

PREREGISTER3 established (on ProntoQA is-a trees) that concept-token reps admit a
low-dimensional hyperbolic decode of the ground-truth tree. The obvious next
question for a GENERAL picture: is that special to taxonomic is-a, or a generic
property of any relational structure the model represents? This module emits the
SAME record schema as ``prontoqa_tree`` (``tree_meta`` with names/parent/depth),
so it flows through ``concept_align`` + ``tree_probe`` unchanged — only the
RELATION verbalised in the premises changes:

  is_a       — "Every X is a Y."            (taxonomic; the PREREGISTER3 baseline)
  part_of    — "Every X is part of a Y."    (meronomy; a different tree relation)
  causes     — "X causes Y."                (a causal DAG rendered as a tree)
  flat_set   — "X is item number k."        (NEGATIVE CONTROL: a flat list, NO
                                             hierarchy -> hyperbolic Δ MUST be ~0.
                                             If it "wins" here the rig manufactures
                                             structure and every positive is void.)

All four use the SAME underlying branching tree shape (so difficulty/token count
match); only the surface relation differs. ``flat_set`` additionally FLATTENS the
target distance to a star (all nodes equidistant from a root) so there is no tree
metric to recover — the sharp negative control.
"""

from __future__ import annotations

import random

from .prontoqa_tree import (ENTITIES, NONCE, _ancestors, _build_tree, _leaves)

RELATIONS = ("is_a", "part_of", "causes", "flat_set")

_PREMISE_TMPL = {
    "is_a": "Every {c} is a {p}.",
    "part_of": "Every {c} is part of a {p}.",
    "causes": "A {c} causes a {p}.",
    "flat_set": "{c} is an item.",   # no relation between items -> flat
}
_QUERY_TMPL = {
    "is_a": "Is it true or false that {e} is a {t}?",
    "part_of": "Is it true or false that {e} is part of a {t}?",
    "causes": "Is it true or false that {e} eventually causes a {t}?",
    "flat_set": "Is it true or false that {e} and {t} are both items?",
}


def _make_one(rng, relation, n_nodes, idx):
    parent, depth = _build_tree(2, n_nodes)         # fixed branching-2 shape for all
    names = rng.sample(NONCE, len(parent))
    n = len(parent)

    premises = [_PREMISE_TMPL[relation].format(c=names[c], p=names[parent[c]])
                for c in range(n) if parent[c] >= 0]
    if relation == "flat_set":
        # every node is just "an item" — no edges verbalised as relations
        premises = [_PREMISE_TMPL[relation].format(c=names[c]) for c in range(n)]
    rng.shuffle(premises)

    leaves = _leaves(parent)
    ent_node = rng.choice(leaves)
    ent = rng.choice(ENTITIES)
    anc = _ancestors(ent_node, parent)
    if idx % 2 == 0 and anc:
        target, answer = rng.choice(sorted(anc)), 1
    else:
        non_anc = [i for i in range(n) if i != ent_node and i not in anc]
        target, answer = (rng.choice(non_anc), 0) if non_anc else (rng.choice(sorted(anc)), 1)

    place = (f"{ent} is a {names[ent_node]}." if relation in ("is_a", "flat_set")
             else f"{ent} is a {names[ent_node]}.")
    prompt = (" ".join(premises) + " " + place + "\n"
              f"Question: {_QUERY_TMPL[relation].format(e=ent, t=names[target])} "
              f"Reason step by step, then answer True or False.")

    # For flat_set the ground-truth "tree" is a STAR: all leaves at distance 2 via
    # a single hub, so there is no graded hierarchy to recover (negative control).
    if relation == "flat_set":
        star_parent = [-1] + [0] * (n - 1)
        star_depth = [0] + [1] * (n - 1)
        tm_parent, tm_depth = star_parent, star_depth
    else:
        tm_parent, tm_depth = parent, depth

    return {
        "sample_id": f"rel_{relation}_{idx}",
        "prompt": prompt,
        "label": RELATIONS.index(relation),
        "label_path": [RELATIONS.index(relation), 0],
        "answer": answer,
        "tree_meta": {
            "branching": 2, "naming": relation, "n_nodes": n,
            "names": names, "parent": tm_parent, "depth": tm_depth,
            "entity": ent, "entity_node": ent_node, "target_node": target,
            "answer": answer, "relation": relation,
        },
    }


def build_relation_trees(n_prompts=60, n_nodes=15, seed=0):
    """All four relation arms in one dataset (is_a / part_of / causes / flat_set)."""
    rows = []
    for ri, rel in enumerate(RELATIONS):
        rng = random.Random(seed * 100 + ri)
        rows += [_make_one(rng, rel, n_nodes, i) for i in range(n_prompts)]
    random.Random(seed).shuffle(rows)
    return rows
