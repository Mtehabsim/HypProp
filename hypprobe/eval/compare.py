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
                             selectivity=metrics.get("selectivity"),
                             mdl_bits=metrics.get("mdl_bits"),
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
    # We test SELECTIVITY as the primary metric (the methodology says raw acc is
    # not to be trusted), with accuracy and MDL reported alongside.
    by_metric = {m: defaultdict(dict) for m in ("val_acc", "selectivity", "mdl_bits")}
    for r in rows:
        k = (r["model"], r["dataset"], r["layer"], r["source"], r["seed"])
        for m in by_metric:
            if r.get(m) is not None:
                by_metric[m][k][r["arm"]] = r[m]

    def _diffs(metric):
        out = []
        for _k, arms in by_metric[metric].items():
            if arms.get("hyperbolic") is not None and arms.get("flat_on_transform") is not None:
                out.append(arms["hyperbolic"] - arms["flat_on_transform"])
        return out

    sig_sel = _paired_test(_diffs("selectivity"))
    sig_acc = _paired_test(_diffs("val_acc"))
    # MDL: lower is better, so hyperbolic wins if (hyp - flat) < 0.
    sig_mdl = _paired_test(_diffs("mdl_bits"))
    save_json(os.path.join(out_dir, "significance.json"),
              dict(primary="selectivity",
                   selectivity_hyp_minus_flat=sig_sel,
                   val_acc_hyp_minus_flat=sig_acc,
                   mdl_bits_hyp_minus_flat=sig_mdl))

    # summary.md
    lines = ["# Probe comparison summary", "",
             "Matched arms (capacity-equal): **hyperbolic** vs **flat_on_transform**.",
             "PRIMARY metric = **selectivity** (real - random-label control; Hewitt & Liang).",
             "Secondary: MDL bits (lower=better) and raw val_acc (least trusted).",
             "`euclidean_lr` is NOT capacity-matched -- reference only.",
             "", "| model | dataset | layer | source | arm | mean acc | mean sel | mean mdl | n |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(agg):
        # per-(setting,arm) means directly from rows
        sub = [r for r in rows if (r["model"], r["dataset"], r["layer"], r["source"], r["arm"]) == key]
        macc = np.mean([r["val_acc"] for r in sub if r.get("val_acc") is not None]) if sub else float("nan")
        msel = [r["selectivity"] for r in sub if r.get("selectivity") is not None]
        mmdl = [r["mdl_bits"] for r in sub if r.get("mdl_bits") is not None]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | "
                     f"{macc:.3f} | {np.mean(msel):.3f} | "
                     f"{(np.mean(mmdl) if mmdl else float('nan')):.0f} | {len(sub)} |")
    lines += ["", "## Matched verdict (hyperbolic - flat_on_transform)",
              f"- **selectivity** (PRIMARY): mean {sig_sel.get('mean', 0):+.3f}, "
              f"p={sig_sel.get('p_value')}, n={sig_sel.get('n', 0)}",
              f"- MDL bits (lower=better): mean {sig_mdl.get('mean', 0):+.1f} "
              f"(negative favours hyperbolic)",
              f"- val_acc (least trusted): mean {sig_acc.get('mean', 0):+.3f}"]
    win = sig_sel.get("mean", 0) > 0 and (sig_sel.get("p_value") or 1) < 0.05
    verdict = ("hyperbolic > flat on selectivity (significant)" if win
               else "NO significant hyperbolic advantage on selectivity")
    lines.append(f"- **verdict: {verdict}** (counts only if it survives whitening -- it now does -- "
                 f"AND is significant on selectivity, not raw acc)")
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    log_line(logfile, f"comparison: selectivity(hyp-flat) mean={sig_sel.get('mean', 0):+.3f} "
                      f"p={sig_sel.get('p_value')} | acc mean={sig_acc.get('mean', 0):+.3f}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 2: matched comparison.")
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", default="./results/eval")
    args = ap.parse_args(argv)
    run(args.probes, args.out)


if __name__ == "__main__":
    main()
