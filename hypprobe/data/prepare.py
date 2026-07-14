"""Phase 0.a: prepare datasets into a uniform sample format.

Each dataset becomes ``<cache>/<name>.jsonl`` with one JSON object per line:
    {"sample_id": str, "prompt": str, "label": int, "label_path": [int, ...]}
where ``label_path`` is the taxonomy path from root to leaf (enables the
hierarchy / tree-distance analysis).

The ``wordnet_control`` set is generated locally (no downloads) so the whole
pipeline can be smoke-tested before touching the real safety corpora. Loaders
for the real datasets (AILuminate, Aegis, HarmBench/AdvBench, WOS) are wired as
clearly-marked stubs that read from a raw drop-in directory on the DGX; they
raise a helpful message if the raw files are absent rather than inventing data.
"""

from __future__ import annotations

import argparse
import json
import os

from ..io import ensure_dir, log_line


def _write_jsonl(path: str, rows: list[dict]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def build_wordnet_control(n_per_leaf: int = 20, seed: int = 0) -> list[dict]:
    """A small, strictly-hierarchical prompt set (no external data needed).

    Classes form a 2-level tree (domain -> subtype). Prompts are templated
    descriptions; the point is a KNOWN tree over labels so the probe pipeline can
    be validated end-to-end. This mirrors the plan's WordNet positive control.
    """
    import random

    rng = random.Random(seed)
    taxonomy = {
        "animal": {"dog": ["a loyal {} guarding the house",
                            "the {} barked at the mail carrier"],
                   "cat": ["a sleepy {} on the windowsill",
                           "the {} chased a laser dot"],
                   "eagle": ["a soaring {} over the canyon",
                             "the {} dove toward the river"]},
        "vehicle": {"car": ["a red {} parked downtown",
                            "the {} merged onto the highway"],
                    "boat": ["a wooden {} at the dock",
                             "the {} drifted across the lake"],
                    "plane": ["a jet {} climbing after takeoff",
                              "the {} taxied to the gate"]},
        "plant": {"oak": ["a towering {} in the meadow",
                          "the {} dropped its acorns"],
                  "rose": ["a red {} in the garden",
                          "the {} bloomed in June"],
                  "fern": ["a green {} in the shade",
                          "the {} unfurled new fronds"]},
    }
    rows = []
    domains = list(taxonomy)
    leaf_id = 0
    for di, dom in enumerate(domains):
        for subtype, templates in taxonomy[dom].items():
            for k in range(n_per_leaf):
                tmpl = rng.choice(templates)
                prompt = tmpl.format(subtype)
                rows.append({
                    "sample_id": f"{dom}_{subtype}_{k}",
                    "prompt": f"Describe: {prompt}",
                    "label": leaf_id,
                    "label_path": [di, leaf_id],
                })
            leaf_id += 1
    rng.shuffle(rows)
    return rows


def build_flat_control(n_per_class: int = 60, seed: int = 0) -> list[dict]:
    """A genuinely FLAT, non-hierarchical binary task (negative control).

    Two classes with NO nested structure (label_path is a single level), so a
    hyperbolic probe should have NO advantage here. If it 'wins' on this set,
    that is a red flag the pipeline is manufacturing hierarchy. This is the
    negative control the plan promised but that previously did not exist.
    """
    import random

    rng = random.Random(seed)
    pos = ["a bright sunny morning", "the cheerful festival crowd",
           "a warm friendly greeting", "the joyful celebration"]
    neg = ["a dull grey afternoon", "the tedious waiting room",
           "a flat monotone lecture", "the empty parking lot"]
    rows = []
    for cls, pool in [(0, neg), (1, pos)]:
        for k in range(n_per_class):
            rows.append({
                "sample_id": f"flat_{cls}_{k}",
                "prompt": f"Describe: {rng.choice(pool)}",
                "label": cls,
                "label_path": [cls],   # single level -> no hierarchy
            })
    rng.shuffle(rows)
    return rows


def build_prontoqa(n_per_depth: int = 60, depths=(1, 2, 3, 4, 5), seed: int = 0) -> list[dict]:
    """Synthetic PrOntoQA (Saparov & He 2023) -- reasoning-eliciting prompts.

    Each example is a nonce-ontology chain: 'Every <A> is a <B>. Every <B> is a
    <C>. ... <X> is a <A>. True or false: <X> is a <Z>?' The nonsense predicates
    ('yumpus', 'wumpus', ...) force the model to actually CHAIN the rules rather
    than recall facts -- so it generates a multi-step reasoning trace, which is
    exactly what WHEN (H1/H2) needs and what a benign 'Describe: ...' prompt does
    not give. This is the dataset Raj used, so our H1/H2 are directly comparable.

    label = reasoning depth (# hops); label_path = a coarse-then-fine path over
    depth so the taxonomy/structural targets have graded structure to recover.
    Fully synthetic: no downloads, deterministic per seed.
    """
    import random

    rng = random.Random(seed)
    # A pool of pronounceable nonce predicates (PrOntoQA-style).
    stems = ["yumpus", "wumpus", "jompus", "zumpus", "numpus", "vumpus",
             "tumpus", "rompus", "dumpus", "sterpus", "lorpus", "grimpus",
             "shumpus", "brimpus", "gorpus", "lempus", "twmpus", "frompus"]
    entities = ["Max", "Alex", "Sam", "Polly", "Rex", "Fae", "Wren", "Stella"]
    rows = []
    for depth in depths:
        for k in range(n_per_depth):
            chain = rng.sample(stems, depth + 1)      # depth edges -> depth+1 kinds
            ent = rng.choice(entities)
            facts = " ".join(f"Every {chain[i]} is a {chain[i+1]}." for i in range(depth))
            true_query = f"{ent} is a {chain[0]}."
            # Half TRUE (ask about the last kind), half FALSE (ask about an unrelated kind).
            if k % 2 == 0:
                target, answer = chain[-1], 1
            else:
                distractor = rng.choice([s for s in stems if s not in chain])
                target, answer = distractor, 0
            prompt = (f"{facts} {true_query}\n"
                      f"Question: Is it true or false that {ent} is a {target}? "
                      f"Reason step by step, then answer True or False.")
            rows.append({
                "sample_id": f"pronto_d{depth}_{k}",
                "prompt": prompt,
                "label": depth,                       # reasoning depth
                "label_path": [0 if depth <= 2 else 1, depth],  # coarse (shallow/deep) -> depth
                "answer": answer,
            })
    rng.shuffle(rows)
    return rows


def _load_real_dataset(name: str, raw_dir: str) -> list[dict]:
    """Load a real dataset from a raw drop-in directory (DGX).

    Expects ``<raw_dir>/<name>.jsonl`` already in the sample schema, OR a known
    raw format we convert. Kept intentionally strict: if the raw file is missing
    we raise, rather than fabricate safety data.
    """
    candidate = os.path.join(raw_dir, f"{name}.jsonl")
    if os.path.exists(candidate):
        rows = []
        with open(candidate) as fh:
            for i, line in enumerate(fh):
                obj = json.loads(line)
                obj.setdefault("sample_id", f"{name}_{i}")
                obj.setdefault("label", 0)
                obj.setdefault("label_path", [obj.get("label", 0)])
                rows.append(obj)
        return rows
    raise FileNotFoundError(
        f"raw data for '{name}' not found at {candidate}. Place the corpus there "
        f"on the DGX (schema: one JSON/line with prompt,label,label_path), or use "
        f"'wordnet_control' for a local smoke test.")


def build_prontoqa_tree_dataset() -> list[dict]:
    """PREREGISTER3 branching-ontology set (fictional b1/b2/b3 + real), tree retained."""
    from .prontoqa_tree import build_prontoqa_tree_all
    return build_prontoqa_tree_all()


def build_relation_trees_dataset() -> list[dict]:
    """Relation-type generality: is_a / part_of / causes / flat_set (negative control)."""
    from .relation_trees import build_relation_trees
    return build_relation_trees()


BUILDERS = {"wordnet_control": build_wordnet_control,
            "flat_control": build_flat_control,
            "prontoqa": build_prontoqa,
            "prontoqa_tree": build_prontoqa_tree_dataset,
            "relation_trees": build_relation_trees_dataset}
REAL = {"ailuminate", "aegis", "harmbench", "advbench", "wos"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prepare datasets (Phase 0.a).")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", default="./results/data_cache")
    ap.add_argument("--raw", default="./raw_data", help="dir with real corpora (DGX)")
    ap.add_argument("--variants", action="store_true",
                    help="also emit nonce/paraphrase variants (meaning control)")
    args = ap.parse_args(argv)

    ensure_dir(args.out)
    logfile = os.path.join(os.path.dirname(args.out.rstrip("/")) or ".",
                           "logs", "prepare.log")
    for ds in args.datasets:
        if ds in BUILDERS:
            rows = BUILDERS[ds]()
        elif ds in REAL:
            try:
                rows = _load_real_dataset(ds, args.raw)
            except FileNotFoundError as exc:
                # Clean, actionable message instead of a scary traceback mid-run.
                raise SystemExit(f"[prepare] {exc}")
        else:
            raise SystemExit(
                f"unknown dataset '{ds}'. Builders: {sorted(BUILDERS)}; "
                f"real (need --raw dir): {sorted(REAL)}")
        out_path = os.path.join(args.out, f"{ds}.jsonl")
        _write_jsonl(out_path, rows)
        n_classes = len({r["label"] for r in rows})
        log_line(logfile, f"prepared {ds}: {len(rows)} samples, {n_classes} classes -> {out_path}")
        if args.variants:
            from .variants import augment_jsonl
            n = augment_jsonl(out_path, out_path)  # in place: original + variants
            log_line(logfile, f"  + variants: {ds} now {n} rows (original+nonce+paraphrase)")


if __name__ == "__main__":
    main()
