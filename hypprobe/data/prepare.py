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


BUILDERS = {"wordnet_control": build_wordnet_control}
REAL = {"ailuminate", "aegis", "harmbench", "advbench", "wos"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prepare datasets (Phase 0.a).")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", default="./results/data_cache")
    ap.add_argument("--raw", default="./raw_data", help="dir with real corpora (DGX)")
    args = ap.parse_args(argv)

    ensure_dir(args.out)
    logfile = os.path.join(os.path.dirname(args.out.rstrip("/")) or ".",
                           "logs", "prepare.log")
    for ds in args.datasets:
        if ds in BUILDERS:
            rows = BUILDERS[ds]()
        elif ds in REAL:
            rows = _load_real_dataset(ds, args.raw)
        else:
            raise SystemExit(f"unknown dataset '{ds}'")
        out_path = os.path.join(args.out, f"{ds}.jsonl")
        _write_jsonl(out_path, rows)
        n_classes = len({r["label"] for r in rows})
        log_line(logfile, f"prepared {ds}: {len(rows)} samples, {n_classes} classes -> {out_path}")


if __name__ == "__main__":
    main()
