"""Harm-detection dataset with a HAZARD TAXONOMY tree (the harm experiment).

The falsifiable prediction from our findings: hyperbolic geometry helps harm
classification IFF the harm label space is HIERARCHICAL. So we build harm prompts
carrying a ground-truth hazard *taxonomy* (supercategory -> category), run the
SAME tree_probe, and compare against a BINARY (safe/unsafe) view of the same
prompts. Prediction: taxonomy view shows the low-dim hyperbolic advantage (like
is_a/WordNet); binary view shows ~0 (like the flat_set control).

Primary source: nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (hierarchical hazard
taxonomy). GATED — needs HF terms acceptance + token. If a 401/absent occurs we
fall back to PKU-Alignment/BeaverTails (public, 14 flat categories) so an
unattended run never stalls; BeaverTails' flatter labels are a weaker hierarchy
test but the binary control is clean either way.

Output schema matches prontoqa_tree so it flows through concept_align + tree_probe
UNCHANGED: each row carries tree_meta with names/parent/depth over the HAZARD
CATEGORIES present in that prompt's neighbourhood. But harm prompts are NOT
per-prompt ontologies (a prompt has ONE hazard label, not a tree of concepts) —
so unlike prontoqa_tree, the "tree" here is the GLOBAL hazard taxonomy and the
decodable unit is the PROMPT'S POOLED representation labelled by its category.
tree_probe's per-prompt concept mode doesn't fit that; instead we use the
label-distance mode (structural_probe._taxonomy_target) via label_path. So the
key output is label_path = [supercategory_id, category_id] (taxonomy) and
answer = 0/1 (binary), and we run the probe in prompt-pooled mode.
"""

from __future__ import annotations

import json
import os

# Aegis 2.0 hazard taxonomy: map its ~13 leaf categories to supercategories so
# label_path = [super, leaf] gives a 2-level tree (the hierarchy under test).
# Grouping follows the MLCommons hazard taxonomy Aegis is built on.
_AEGIS_SUPER = {
    # violence & physical harm
    "Violence": "PhysicalHarm", "Weapons": "PhysicalHarm",
    "Criminal Planning/Confessions": "PhysicalHarm",
    "Threat": "PhysicalHarm",
    # sexual
    "Sexual": "Sexual", "Sexual (minor)": "Sexual",
    # self-directed
    "Suicide and Self Harm": "SelfHarm",
    # societal / rights
    "Hate/Identity Hate": "SocietalHarm", "Harassment": "SocietalHarm",
    "PII/Privacy": "SocietalHarm", "Illegal Activity": "SocietalHarm",
    # information hazards
    "Guns and Illegal Weapons": "PhysicalHarm",
    "Controlled/Regulated Substances": "PhysicalHarm",
    "Fraud/Deception": "SocietalHarm", "Malware": "SocietalHarm",
    "Profanity": "SocietalHarm", "Other": "SocietalHarm",
}
_SUPERS = ["PhysicalHarm", "Sexual", "SelfHarm", "SocietalHarm", "Safe"]


def _prompt_text(text: str) -> str:
    """Wrap a raw harmful/benign prompt in a neutral classification framing so the
    model forms a representation of it (we read prompt-side reps, not a generation)."""
    text = (text or "").strip().replace("\n", " ")
    if len(text) > 600:
        text = text[:600]
    return f"Consider the following request:\n\"{text}\"\nAssess its safety."


# Stable leaf ids so prompts in the SAME category share a leaf (tree distance 0),
# same-supercategory differ at the leaf (distance 2), cross-supercategory differ
# at the root (distance 4). A random/hashed id would make every prompt its own
# leaf -> a degenerate star with no gradation (the taxonomy view would be flat and
# the whole hierarchy-vs-binary contrast meaningless). Verified via _taxonomy_target.
_LEAF_IDS: dict[str, int] = {}


def _leaf_id(leafcat: str) -> int:
    if leafcat not in _LEAF_IDS:
        _LEAF_IDS[leafcat] = len(_LEAF_IDS)
    return _LEAF_IDS[leafcat]


def _safe_id(sid: str) -> str:
    """Filesystem-safe sample_id: category names contain '/' and spaces (e.g.
    'Controlled/Regulated Substances') which broke sample_path (it built a
    nonexistent nested dir and the extractor crashed). Collapse to [A-Za-z0-9_-]."""
    out = []
    for ch in str(sid):
        out.append(ch if (ch.isalnum() or ch in "_-") else "_")
    return "".join(out)


def _row(sid, text, supercat, leafcat, answer):
    supers = _SUPERS
    sup_id = supers.index(supercat) if supercat in supers else len(supers) - 1
    return {
        "sample_id": _safe_id(sid),
        "prompt": _prompt_text(text),
        "label": answer,                    # binary safe(0)/unsafe(1) — the flat view
        "label_path": [sup_id, _leaf_id(leafcat)],  # [super, leaf] — the taxonomy view
        "answer": answer,
        "harm_supercategory": supercat,
        "harm_category": leafcat,
    }


def _from_aegis(n_per_cat=40, seed=0):
    """Load Aegis 2.0 from HF. Raises on gated/absent so caller can fall back."""
    from datasets import load_dataset
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
    import random
    rng = random.Random(seed)
    rows, by_cat = [], {}
    for ex in ds:
        # Aegis fields: 'prompt', 'prompt_label' (safe/unsafe), 'violated_categories'
        text = ex.get("prompt") or ex.get("text") or ""
        label = (ex.get("prompt_label") or ex.get("label") or "").lower()
        answer = 1 if "unsafe" in label else 0
        cats = ex.get("violated_categories") or ex.get("category") or ""
        leaf = (cats.split(",")[0].strip() if isinstance(cats, str) else
                (cats[0] if cats else "Other")) or "Other"
        if answer == 0:
            supercat, leaf = "Safe", "Safe"
        else:
            supercat = _AEGIS_SUPER.get(leaf, "SocietalHarm")
        by_cat.setdefault((supercat, leaf), []).append((text, answer))
    for (supercat, leaf), items in by_cat.items():
        rng.shuffle(items)
        for i, (text, answer) in enumerate(items[:n_per_cat]):
            rows.append(_row(f"aegis_{supercat}_{leaf}_{i}", text, supercat, leaf, answer))
    rng.shuffle(rows)
    return rows, "aegis2.0"


def _from_beavertails(n_per_cat=40, seed=0):
    """Public fallback: BeaverTails (14 flat harm categories + is_safe)."""
    from datasets import load_dataset
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    import random
    rng = random.Random(seed)
    rows, by_cat = [], {}
    for ex in ds:
        text = ex.get("prompt") or ""
        cats = ex.get("category") or {}
        is_safe = bool(ex.get("is_safe", False))
        answer = 0 if is_safe else 1
        if is_safe:
            supercat, leaf = "Safe", "Safe"
        else:
            # BeaverTails category is a dict {name: bool}; take first True
            true_cats = [k for k, v in cats.items() if v] if isinstance(cats, dict) else []
            leaf = true_cats[0] if true_cats else "Other"
            # coarse grouping of BeaverTails categories into supercategories
            leaf_l = leaf.lower()
            if any(w in leaf_l for w in ("violence", "weapon", "crime", "terror", "abuse")):
                supercat = "PhysicalHarm"
            elif any(w in leaf_l for w in ("sexual", "porn", "child")):
                supercat = "Sexual"
            elif any(w in leaf_l for w in ("self", "suicide")):
                supercat = "SelfHarm"
            else:
                supercat = "SocietalHarm"
        by_cat.setdefault((supercat, leaf), []).append((text, answer))
    for (supercat, leaf), items in by_cat.items():
        rng.shuffle(items)
        for i, (text, answer) in enumerate(items[:n_per_cat]):
            rows.append(_row(f"bt_{supercat}_{leaf}_{i}", text, supercat, leaf, answer))
    rng.shuffle(rows)
    return rows, "beavertails"


def build_harm_taxonomy(n_per_cat=40, seed=0) -> list[dict]:
    """Aegis 2.0 if accessible, else BeaverTails. Records the source used."""
    try:
        rows, src = _from_aegis(n_per_cat=n_per_cat, seed=seed)
        print(f"[harm_taxonomy] source=aegis2.0 rows={len(rows)}", flush=True)
        return rows
    except Exception as exc:   # gated (401), no token, or schema drift
        print(f"[harm_taxonomy] Aegis unavailable ({type(exc).__name__}: "
              f"{str(exc)[:120]}); falling back to BeaverTails", flush=True)
        rows, src = _from_beavertails(n_per_cat=n_per_cat, seed=seed)
        print(f"[harm_taxonomy] source=beavertails rows={len(rows)}", flush=True)
        return rows
