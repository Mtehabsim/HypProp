"""Phase 2 CLI: assemble the matched cross-arm comparison.

Reads all per-arm metrics json files, aggregates across seeds, and emits:
  - comparison.csv        : every probe x setting x seed
  - summary.md            : human-readable table with the MATCHED verdict
  - significance.json     : paired difference (hyperbolic vs flat_on_transform)

The headline comparison is hyperbolic vs flat_on_transform, because those are
capacity-matched (same proj_dim, same transform, only geometry differs). We flag
euclidean_lr separately since it uses full-dim features (not capacity matched).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from ..io import ensure_dir, log_line, save_csv, save_json


def _load_all(probes_dir):
    rows = []
    for path in glob.glob(os.path.join(probes_dir, "*.json")):
        with open(path) as fh:
            obj = json.load(fh)
        if "arms" not in obj:
            continue
        for arm, metrics in obj["arms"].items():
            rows.append(dict(model=obj.get("model"), dataset=obj.get("dataset"),
                             layer=obj.get("layer"), source=obj.get("source"),
                             seed=obj.get("seed"), proj_dim=obj.get("proj_dim"),
                             arm=arm, val_acc=metrics.get("val_acc"),
                             macro_f1=metrics.get("macro_f1"),
                             curvature=metrics.get("curvature", "")))
    return rows


def _paired_test(diffs):
    """Simple paired t-like summary: mean, std, and a normal-approx p-value."""
    diffs = np.asarray([d for d in diffs if d is not None], float)
    if len(diffs) < 2:
        return dict(n=len(diffs), mean=float(diffs.mean()) if len(diffs) else 0.0,
                    p_value=None)
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    if se == 0:
        p = 0.0 if mean != 0 else 1.0
    else:
        from math import erf, sqrt
        z = abs(mean) / se
        p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    return dict(n=len(diffs), mean=mean, se=se, p_value=p)


def run(probes_dir, out_dir):
    ensure_dir(out_dir)
    logfile = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "logs", "compare.log")
    rows = _load_all(probes_dir)
    save_csv(os.path.join(out_dir, "comparison.csv"), rows,
             columns=["model", "dataset", "layer", "source", "seed", "proj_dim",
                      "arm", "val_acc", "macro_f1", "curvature"])

    # Aggregate per (setting, arm).
    agg = defaultdict(list)
    for r in rows:
        key = (r["model"], r["dataset"], r["layer"], r["source"], r["arm"])
        if r["val_acc"] is not None:
            agg[key].append(r["val_acc"])

    # Matched paired diffs: hyperbolic - flat_on_transform, same setting+seed.
    by_setting_seed = defaultdict(dict)
    for r in rows:
        k = (r["model"], r["dataset"], r["layer"], r["source"], r["seed"])
        by_setting_seed[k][r["arm"]] = r["val_acc"]
    diffs = []
    for k, arms in by_setting_seed.items():
        if "hyperbolic" in arms and "flat_on_transform" in arms:
            if arms["hyperbolic"] is not None and arms["flat_on_transform"] is not None:
                diffs.append(arms["hyperbolic"] - arms["flat_on_transform"])
    sig = _paired_test(diffs)
    save_json(os.path.join(out_dir, "significance.json"),
              dict(comparison="hyperbolic - flat_on_transform (matched)", **sig))

    # summary.md
    lines = ["# Probe comparison summary", "",
             "Matched arms (capacity-equal): **hyperbolic** vs **flat_on_transform**.",
             "`euclidean_lr` uses full-dim features (NOT capacity-matched) -- shown for reference.",
             "", "| model | dataset | layer | source | arm | mean val_acc | n |",
             "|---|---|---|---|---|---|---|"]
    for key in sorted(agg):
        accs = agg[key]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | "
                     f"{np.mean(accs):.3f} | {len(accs)} |")
    lines += ["", "## Matched verdict (hyperbolic - flat_on_transform)",
              f"- mean difference: {sig.get('mean', 0):+.3f} over n={sig.get('n', 0)} settings",
              f"- p-value: {sig.get('p_value')}"]
    verdict = ("hyperbolic > flat" if sig.get("mean", 0) > 0 else "no hyperbolic advantage")
    lines.append(f"- **verdict: {verdict}** "
                 f"(remember: only counts if it survives whitening + is significant)")
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    log_line(logfile, f"comparison: hyperbolic - flat_on_transform mean={sig.get('mean', 0):+.3f} "
                      f"(n={sig.get('n', 0)}, p={sig.get('p_value')})")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2: matched comparison.")
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", default="./results/eval")
    args = ap.parse_args(argv)
    run(args.probes, args.out)


if __name__ == "__main__":
    main()
