"""WHEN — score H1/H2/H3 (the Raj <-> Atlas bridge) from the Rung-0 output.

Rung 0 already emits, per (model, layer, token_source, metric), the delta_rel and
its noise floor. This module *scores* the pre-registered WHEN hypotheses off that
same `rung0.csv` — so the DGX run returns both the Rung-0 verdict AND the WHEN
verdict in one shot, against thresholds committed in PREREGISTER.md before looking.

Hypotheses (measured under the `background` metric, at the final layer):
  H1 (generation amplifies compression):
      delta_final(generated) < delta_final(prompt) by > h1_margin and > k*floor.
  H2 (explains Raj's model-specificity):
      gap(reasoning_model) - gap(base_model) > h2_gap_margin,
      where gap(m) = delta_final(prompt) - delta_final(generated) for model m.
  H3 (locus, secondary/reported):
      thinking-marker tokens are the lowest-delta source at the final layer.

Everything is read, not recomputed, so this is CPU-instant and deterministic.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from ..io import ensure_dir, log_line

# Which model ids count as "reasoning-tuned" for H2. Substring match, lowercase.
REASONING_MARKERS = ("r1", "distill", "reason", "deepseek-r1")

DEFAULT_WHEN = {
    "h1_margin": 0.05,
    "h1_min_effect_over_boot": 2.0,
    "h2_gap_margin": 0.05,
    "max_bootstrap_std_final": 0.02,   # power target: floor must be below this
}


def _load_when_thresholds(project_root):
    path = os.path.join(project_root, "PREREGISTER.md")
    if os.path.exists(path):
        txt = open(path).read()
        if "```json" in txt:
            block = txt.split("```json", 1)[1].split("```", 1)[0]
            try:
                th = json.loads(block).get("when", {})
                merged = dict(DEFAULT_WHEN)
                merged.update(th)
                return merged, "PREREGISTER.md"
            except Exception:
                pass
    return dict(DEFAULT_WHEN), "defaults"


def _final_layer_delta(rows, model, source, metric="background"):
    """(delta_rel, noise_floor) at the final (max-index) layer, or None."""
    sub = [r for r in rows if r["model"] == model and r["token_source"] == source
           and r["metric"] == metric and r["cloud_kind"] == "data"]
    if not sub:
        return None
    final = max(sub, key=lambda r: int(r["layer"]))
    return float(final["delta_rel"]), float(final["noise_floor"])


def _is_reasoning(model):
    m = model.lower()
    return any(k in m for k in REASONING_MARKERS)


def score(rows, thresholds):
    """Return a dict of H1/H2/H3 results scored against the thresholds."""
    models = sorted({r["model"] for r in rows if r["cloud_kind"] == "data"})
    out = {"h1": [], "h2": None, "h3": [], "gaps": {}}

    # --- H1 per model: generated more tree-like than prompt at final layer? ---
    for model in models:
        prompt = _final_layer_delta(rows, model, "input")
        gen = _final_layer_delta(rows, model, "generated")
        if prompt is None or gen is None:
            continue
        d_prompt, f_p = prompt
        d_gen, f_g = gen
        diff = d_prompt - d_gen                     # >0 means generated is MORE tree-like
        floor = max(f_p, f_g, 1e-6)
        # Power: is the noise floor small enough that an AMBIGUOUS call is
        # trustworthy? If floor > target, an AMBIGUOUS/near-miss is likely
        # UNDERPOWERED (add prompts), not a real null.
        target = thresholds.get("max_bootstrap_std_final", 0.02)
        underpowered = floor > target
        if diff > thresholds["h1_margin"] and diff > thresholds["h1_min_effect_over_boot"] * floor:
            verdict = "PASS"
        elif diff <= 0 and not underpowered:
            verdict = "FAIL"
        elif diff <= 0 and underpowered:
            verdict = "FAIL?(underpowered)"
        elif underpowered:
            verdict = "AMBIGUOUS(underpowered)"
        else:
            verdict = "AMBIGUOUS(powered)"
        out["h1"].append(dict(model=model, delta_prompt=round(d_prompt, 4),
                              delta_generated=round(d_gen, 4), diff=round(diff, 4),
                              floor=round(floor, 4), target=target,
                              underpowered=underpowered, verdict=verdict,
                              reasoning=_is_reasoning(model)))
        out["gaps"][model] = diff

    # --- H2: is the prompt->generated gap larger for reasoning models? ---
    reasoning_gaps = [g for m, g in out["gaps"].items() if _is_reasoning(m)]
    base_gaps = [g for m, g in out["gaps"].items() if not _is_reasoning(m)]
    if reasoning_gaps and base_gaps:
        import numpy as np
        gap_r = float(np.mean(reasoning_gaps))
        gap_b = float(np.mean(base_gaps))
        margin = gap_r - gap_b
        out["h2"] = dict(gap_reasoning=round(gap_r, 4), gap_base=round(gap_b, 4),
                         margin=round(margin, 4),
                         verdict="PASS" if margin > thresholds["h2_gap_margin"] else "FAIL/AMBIGUOUS")

    # --- H3: is 'thinking' the lowest-delta source at the final layer? ---
    for model in models:
        finals = {}
        for src in ("input", "generated", "thinking", "last", "all"):
            v = _final_layer_delta(rows, model, src)
            if v is not None:
                finals[src] = v[0]
        if finals:
            lowest = min(finals, key=finals.get)
            out["h3"].append(dict(model=model, lowest_source=lowest,
                                  thinking_is_lowest=(lowest == "thinking"),
                                  finals={k: round(v, 4) for k, v in finals.items()}))
    return out


def _render(result, thresholds, th_source):
    L = ["# WHEN verdict (H1/H2/H3 — the Raj <-> Atlas bridge)", "",
         f"Thresholds source: **{th_source}**  ", f"`{json.dumps(thresholds)}`",
         "", "Measured under the `background` metric at the final layer.", ""]

    L += ["## H1 — does generation amplify the final-layer compression?",
          "(PASS = generated is more tree-like than prompt, beyond margin and noise)", ""]
    if result["h1"]:
        for r in result["h1"]:
            L.append(f"- **{r['model']}**: δ(prompt)={r['delta_prompt']} vs "
                     f"δ(generated)={r['delta_generated']} → diff={r['diff']} "
                     f"(floor {r['floor']}, target {r['target']}) → **{r['verdict']}**")
        # Power banner: an underpowered AMBIGUOUS is NOT evidence of no effect.
        under = [r for r in result["h1"] if r["underpowered"]]
        if under:
            worst = max(r["floor"] for r in under)
            L += ["",
                  f"> ⚠ **POWER WARNING:** final-layer bootstrap_std up to {round(worst,4)} "
                  f"exceeds the pre-registered target {result['h1'][0]['target']}. Any "
                  f"AMBIGUOUS/FAIL above is likely UNDERPOWERED, not a real null — "
                  f"**add prompts and re-run** before concluding. Do NOT read an "
                  f"underpowered result as 'no effect' (this is the replan-#4 trap)."]
    else:
        L.append("- no (prompt, generated) pairs found — was generation extracted?")

    L += ["", "## H2 — is the prompt→generated gap larger in reasoning-tuned models?",
          "(the mechanism Raj never gave for DeepSeek-vs-Qwen)", ""]
    if result["h2"]:
        h2 = result["h2"]
        L.append(f"- gap(reasoning)={h2['gap_reasoning']} vs gap(base)={h2['gap_base']} "
                 f"→ margin={h2['margin']} → **{h2['verdict']}**")
    else:
        L.append("- need >=1 reasoning model AND >=1 base model with both token sources.")

    L += ["", "## H3 — is the thinking-token source the lowest-δ locus? (secondary)", ""]
    for r in result["h3"]:
        L.append(f"- **{r['model']}**: lowest-δ source = `{r['lowest_source']}` "
                 f"(thinking lowest: {r['thinking_is_lowest']}) — {r['finals']}")

    # Overall one-liner. Distinguish a POWERED null (real "drop the framing")
    # from an UNDERPOWERED soft result (add prompts, do NOT conclude).
    h1_pass = [r for r in result["h1"] if r["verdict"] == "PASS"]
    any_underpowered = any(r["underpowered"] for r in result["h1"])
    powered_null = result["h1"] and not h1_pass and not any_underpowered
    L += ["", "## Summary"]
    if h1_pass and result["h2"] and result["h2"]["verdict"] == "PASS":
        L.append("- **H1 + H2 PASS → the generation-amplifies-compression bridge holds.** "
                 "Raj's model-specific effect is explained; proceed with the WHEN framing.")
    elif any_underpowered and not h1_pass:
        L.append("- **UNDERPOWERED → inconclusive, NOT a null.** The floor exceeds the "
                 "pre-registered target on at least one model, so H1 cannot be decided. "
                 "**Add prompts and re-run** — do not drop the framing yet.")
    elif powered_null:
        L.append("- **H1 does not pass and the run IS powered → genuinely drop the "
                 "'generation amplifies' framing.** Raj's effect is not about generation.")
    else:
        L.append("- **Partial** → report per-model; do not over-claim the bridge.")
    return "\n".join(L) + "\n"


def run(geometry_dir, project_root="."):
    """Read rung0.csv from geometry_dir, score WHEN, write when_verdict.md."""
    csv_path = os.path.join(geometry_dir, "rung0.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found — run rung0 first.")
    rows = list(csv.DictReader(open(csv_path)))
    thresholds, th_source = _load_when_thresholds(project_root)
    result = score(rows, thresholds)
    ensure_dir(geometry_dir)
    text = _render(result, thresholds, th_source)
    with open(os.path.join(geometry_dir, "when_verdict.md"), "w") as fh:
        fh.write(text)
    logfile = os.path.join(os.path.dirname(geometry_dir.rstrip("/")) or ".", "logs", "when.log")
    log_line(logfile, "WHEN verdict written to when_verdict.md")
    print("\n" + text)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score WHEN H1/H2/H3 from rung0.csv.")
    ap.add_argument("--geometry", default="./results/geometry")
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args(argv)
    run(args.geometry, project_root=args.project_root)


if __name__ == "__main__":
    main()
