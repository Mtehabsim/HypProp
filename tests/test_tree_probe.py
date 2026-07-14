"""Tests for concept alignment + the ground-truth-tree distortion probe.

These codify the invariants verified on the CPU positive control:
  * concept->token alignment locates every concept on BPE-marked tokens;
  * the shared decoder recovers a genuine branching tree and hyperbolic beats
    matched-capacity Euclidean at low-mid dimension (the positive control);
  * the radial norm<->depth signal lives in RAW features and is destroyed by
    whitening (why the run reads it pre-whitening);
  * a noise layer yields ~0 advantage (negative control);
  * shuffled-tree targets collapse the score.
"""

import numpy as np
import pytest
import torch

torch.set_num_threads(2)


# --------------------------------------------------------------------------- #
# Concept alignment
# --------------------------------------------------------------------------- #
def _mock_store(tmp_path, **kw):
    from hypprobe.data.mock_activations import generate_prontoqa_tree
    out = str(tmp_path / "act")
    generate_prontoqa_tree(out, n_prompts=kw.get("n_prompts", 16),
                           n_nodes=kw.get("n_nodes", 15),
                           n_layers=kw.get("n_layers", 8),
                           hidden=kw.get("hidden", 48),
                           signal_layer=kw.get("signal_layer", 5),
                           noise=kw.get("noise", 0.15), seed=0)
    return out


def test_find_concept_spans_bpe():
    """Sliding-window match locates a multi-sub-token concept on Ġ-marked tokens."""
    from hypprobe.geometry.concept_align import find_concept_spans
    tokens = ["ĠEvery", "Ġwum", "pus", "Ġis", "Ġa", "Ġjom", "pus", "."]
    spans = find_concept_spans(tokens, ["wumpus", "jompus"])
    assert spans["wumpus"] == [(1, 3)]                 # 'Ġwum' + 'pus'
    assert spans["jompus"] == [(5, 7)]


def test_find_concept_spans_word_boundary():
    """A short name must not match inside a longer token/word."""
    from hypprobe.geometry.concept_align import find_concept_spans
    tokens = ["Ġthe", "Ġcategory", "Ġcat", "."]
    spans = find_concept_spans(tokens, ["cat"])
    assert spans["cat"] == [(2, 3)]                    # only the standalone 'Ġcat'


def test_align_sample_locates_concepts(tmp_path):
    from hypprobe.io import iter_samples
    from hypprobe.geometry.concept_align import align_sample
    out = _mock_store(tmp_path)
    frac = []
    for s in iter_samples(out, dataset="prontoqa_tree"):
        al = align_sample(s, role="premise", pool="last")
        frac.append(al["n_matched"] / max(al["n_nodes"], 1))
    assert np.mean(frac) > 0.9                          # ~all concepts located


def test_concept_matrix_shapes(tmp_path):
    from hypprobe.io import iter_samples
    from hypprobe.geometry.concept_align import concept_matrix
    out = _mock_store(tmp_path)
    s = next(iter_samples(out, dataset="prontoqa_tree"))
    X, ids, D, depths = concept_matrix(s, 5, role="premise")
    assert X.shape[0] == len(ids) == D.shape[0] == len(depths)
    assert D.shape == (X.shape[0], X.shape[0])
    assert np.allclose(np.diag(D), 0)                   # tree self-distance is 0


# --------------------------------------------------------------------------- #
# Tree probe — positive & negative controls
# --------------------------------------------------------------------------- #
def _prep(out, layer, role="premise"):
    from hypprobe.io import iter_samples
    from hypprobe.geometry.concept_align import concept_matrix
    from hypprobe.geometry.matched_probe import _whiten_fit
    g = []
    for s in iter_samples(out, dataset="prontoqa_tree"):
        cm = concept_matrix(s, layer, role=role, pool="last")
        if cm is not None:
            g.append(cm)
    Xs = [x[0] for x in g]
    Ds = [x[2] for x in g]
    deps = [x[3] for x in g]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xs))
    ntr = int(0.7 * len(Xs))
    tr, va = perm[:ntr], perm[ntr:]
    wf = _whiten_fit(np.concatenate([Xs[i] for i in tr]))
    trp = [(wf(Xs[i]), Ds[i]) for i in tr]
    vap = [(wf(Xs[i]), Ds[i]) for i in va]
    raw_val = [(Xs[i], deps[i]) for i in va]
    return trp, vap, raw_val


def test_decoder_recovers_tree_positive_control(tmp_path):
    """On the signal layer the shared decoder recovers the branching tree."""
    from hypprobe.geometry.tree_probe import fit_tree_arm
    out = _mock_store(tmp_path, n_prompts=24)
    trp, vap, _ = _prep(out, layer=5)
    res = fit_tree_arm("hyperbolic", trp, vap, proj_dim=5, seed=0,
                       max_epochs=400, check_every=50, patience=5)
    assert res["val_rho"] > 0.2, f"decoder failed on positive control: {res['val_rho']}"


def test_hyperbolic_beats_euclidean_at_low_dim(tmp_path):
    """Positive control: at low-mid dim, hyperbolic > matched-capacity Euclidean.
    If this fails the instrument is broken, not the hypothesis."""
    from hypprobe.geometry.tree_probe import fit_tree_arm
    out = _mock_store(tmp_path, n_prompts=24)
    trp, vap, _ = _prep(out, layer=5)
    euc = fit_tree_arm("cond_euclidean", trp, vap, proj_dim=5, seed=0,
                       max_epochs=400, check_every=50, patience=5)["val_rho"]
    hyp = fit_tree_arm("hyperbolic", trp, vap, proj_dim=5, seed=0,
                       max_epochs=400, check_every=50, patience=5)["val_rho"]
    assert hyp > euc, f"hyperbolic ({hyp:.3f}) should beat euclidean ({euc:.3f}) at m=5"


def test_shuffled_tree_collapses(tmp_path):
    """Permuting the target tree distances collapses the score to ~0."""
    from hypprobe.geometry.tree_probe import fit_tree_arm, _per_prompt_rho
    out = _mock_store(tmp_path, n_prompts=24)
    trp, vap, _ = _prep(out, layer=5)
    res = fit_tree_arm("hyperbolic", trp, vap, proj_dim=5, seed=0,
                       max_epochs=400, check_every=50, patience=5)
    shuf = _per_prompt_rho(res["model"], vap,
                           shuffle_rng=np.random.default_rng(1000))
    assert abs(shuf) < 0.15, f"shuffled-tree null should be ~0, got {shuf:.3f}"


def test_radial_signal_in_raw_not_whitened(tmp_path):
    """Radial norm<->depth lives in RAW features; whitening erases it.
    This is why the run reads the radial fingerprint before whitening."""
    from hypprobe.geometry.tree_probe import radial_depth_correlation
    from hypprobe.geometry.matched_probe import _whiten_fit
    out = _mock_store(tmp_path, n_prompts=24)
    _, _, raw_val = _prep(out, layer=5)
    raw_rho, _, _ = radial_depth_correlation(raw_val)
    # whiten the same features and re-measure
    Xcat = np.concatenate([X for X, _ in raw_val])
    wf = _whiten_fit(Xcat)
    wh_val = [(wf(X), d) for X, d in raw_val]
    wh_rho, _, _ = radial_depth_correlation(wh_val)
    assert raw_rho > 0.4, f"raw radial signal should be strong, got {raw_rho:.3f}"
    assert raw_rho > wh_rho + 0.2, "whitening should visibly weaken the radial signal"


def test_noise_layer_no_advantage(tmp_path):
    """Negative control: a noise layer yields ~0 recovery and no hyp advantage."""
    from hypprobe.geometry.tree_probe import fit_tree_arm
    out = _mock_store(tmp_path, n_prompts=24)
    trp, vap, _ = _prep(out, layer=1)               # noise layer
    hyp = fit_tree_arm("hyperbolic", trp, vap, proj_dim=5, seed=0,
                       max_epochs=300, check_every=50, patience=4)["val_rho"]
    assert abs(hyp) < 0.2, f"noise layer should not recover a tree, got {hyp:.3f}"


def test_curvature_zero_matches_euclidean_arm(tmp_path):
    """c->0 hyperbolic fit ~ cond_euclidean fit (the fairness identity)."""
    from hypprobe.geometry.tree_probe import fit_tree_arm
    out = _mock_store(tmp_path, n_prompts=20)
    trp, vap, _ = _prep(out, layer=5)
    hyp0 = fit_tree_arm("hyperbolic", trp, vap, proj_dim=5, seed=0, curvature=1e-9,
                        max_epochs=300, check_every=50, patience=4)["val_rho"]
    euc = fit_tree_arm("cond_euclidean", trp, vap, proj_dim=5, seed=0,
                       max_epochs=300, check_every=50, patience=4)["val_rho"]
    assert abs(hyp0 - euc) < 0.12, f"c->0 ({hyp0:.3f}) should ~ euclidean ({euc:.3f})"
