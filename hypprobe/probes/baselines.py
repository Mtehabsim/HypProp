"""Fair baseline probes and the matched-comparison harness.

Every arm shares the same features and (where relevant) the same projection, so
any difference is attributable to geometry, not capacity:

  - ``euclidean_lr``      : sklearn LogisticRegression on whitened features
                            (the classic flat linear probe / original draft).
  - ``flat_on_transform`` : H-MLR with use_manifold=False -- same learnable
                            projection, Euclidean decision boundary. Isolates the
                            projection nonlinearity from curvature.
  - ``curvature_zero``    : H-MLR with c=0. Numerically must match a flat probe.
  - ``hyperbolic``        : H-MLR with c>0.

Also provides :func:`online_codelength` for MDL, the sample-efficiency measure
(Voita & Titov 2020): a more credible "beats baseline" signal than raw accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hmlr import ProbeConfig, fit_probe


def whiten_fit(x: np.ndarray, eps: float = 1e-6, pca_cap: int = 256):
    """Fit a leakage-safe PCA-then-whiten transform on ``x``.

    Returns ``(transform_fn, mean, W)`` where ``transform_fn`` maps any matrix
    into the same k-dim whitened space (k = min(N//3, pca_cap, d)). This is the
    SAME sound recipe used for delta (PCA before whitening) so we do not invert
    near-null eigenvalues in the N<<d regime. Fit on TRAIN only, apply to val.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    k = max(1, min(n // 3, pca_cap, d))
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(k, s.shape[0])
    comps = vt[:k]                                  # (k, d)
    std = np.clip(s[:k] / np.sqrt(max(n - 1, 1)), eps, None)
    w = comps.T / std[None, :]                      # (d, k): project + scale

    def transform(z: np.ndarray) -> np.ndarray:
        return (np.asarray(z, dtype=np.float64) - mu) @ w

    return transform, mu, w


def prepare_features(x_train, x_val, whiten: bool = True):
    """Shared feature prep for ALL probe arms: fit whitening on train, apply to
    both, so the matched hyperbolic-vs-flat comparison runs on identical whitened
    low-rank features (this is the confound the project exists to control)."""
    if not whiten:
        return np.asarray(x_train, float), np.asarray(x_val, float)
    tf, _, _ = whiten_fit(x_train)
    return tf(x_train), tf(x_val)


def euclidean_lr(x_train, y_train, x_val, y_val, seed: int = 0, prewhitened: bool = False):
    """Flat logistic-regression probe (sklearn).

    If ``prewhitened`` the caller already whitened (shared with the other arms);
    otherwise we whiten here so a standalone call still gets the fair baseline.
    """
    from sklearn.linear_model import LogisticRegression

    if prewhitened:
        xt, xv = np.asarray(x_train, float), np.asarray(x_val, float)
    else:
        tf, _, _ = whiten_fit(x_train)
        xt, xv = tf(x_train), tf(x_val)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(xt, y_train)
    pred = clf.predict(xv)
    acc = float(np.mean(pred == y_val))
    return clf, acc


def online_codelength(
    x: np.ndarray,
    y: np.ndarray,
    cfg: ProbeConfig,
    n_blocks: int = 8,
) -> float:
    """Prequential (online) MDL codelength in bits (Voita & Titov 2020).

    Train on a growing prefix, score the codelength of the next block. Lower =
    the representation encodes the labels more efficiently. This rewards
    sample-efficiency, so a hyperbolic probe that needs fewer examples wins here
    even when final accuracy ties.
    """
    import torch
    import torch.nn.functional as F

    from .hmlr import HyperbolicMLR

    n = x.shape[0]
    order = np.random.default_rng(cfg.seed).permutation(n)
    x, y = x[order], y[order]
    bounds = np.linspace(0, n, n_blocks + 1, dtype=int)
    total_bits = 0.0
    dev = torch.device(cfg.device)
    # First block: uniform codelength.
    first = bounds[1] - bounds[0]
    total_bits += first * np.log2(cfg.n_classes)
    for b in range(1, n_blocks):
        tr_end = bounds[b]
        te_end = bounds[b + 1]
        model = HyperbolicMLR(cfg).to(dev)
        # quick fit on prefix
        _fit_quick(model, x[:tr_end], y[:tr_end], cfg)
        model.eval()
        with torch.no_grad():
            logits = model(torch.as_tensor(x[tr_end:te_end], dtype=torch.float32, device=dev))
            logp = F.log_softmax(logits, dim=-1).cpu().numpy()
        yb = y[tr_end:te_end]
        total_bits += float(-np.sum(logp[np.arange(len(yb)), yb]) / np.log(2))
    return total_bits


def evaluate_arm(xt, ytr, xv, yva, cfg, control_seed=1234):
    """Return (val_acc, selectivity, mdl_bits) for one arm on shared features.

    - val_acc: real-task validation accuracy.
    - selectivity (Hewitt & Liang 2019): real_acc - control_acc, where the
      control task fits the SAME probe on RANDOMLY PERMUTED labels. High
      selectivity means the probe reads real structure, not just memorises;
      it is the metric the methodology says to trust over raw accuracy.
    - mdl_bits (Voita & Titov 2020): prequential codelength; lower = more
      sample-efficient encoding of the labels.
    """
    from .hmlr import fit_probe as _fit

    _, real = _fit(xt, ytr, xv, yva, cfg)
    # Control task: shuffle the training labels (fixed control seed).
    rng = np.random.default_rng(control_seed)
    y_ctrl = rng.permutation(ytr)
    _, ctrl = _fit(xt, y_ctrl, xv, yva, cfg)
    selectivity = real.val_acc - ctrl.val_acc
    mdl = online_codelength(np.asarray(xt, float), np.asarray(ytr, int), cfg)
    return real.val_acc, real.macro_f1, float(selectivity), float(mdl), real.curvature


def _fit_quick(model, x, y, cfg):
    import torch
    import torch.nn as nn

    dev = torch.device(cfg.device)
    xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.long, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(min(cfg.epochs, 100)):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()


@dataclass
class ArmResult:
    name: str
    val_acc: float
    macro_f1: float
    curvature: float


def run_all_arms(
    x_train, y_train, x_val, y_val, in_dim, n_classes,
    proj_dim: int = 5, curvature: float = 1.0, seed: int = 0, epochs: int = 200,
    whiten: bool = True,
) -> list[ArmResult]:
    """Train every arm on the SAME whitened features and return their metrics.

    Whitening is fit on train and applied to both, ONCE, then shared by all arms
    -- so hyperbolic vs flat_on_transform differ only in geometry, not in whether
    the compression/anisotropy confound was removed.
    """
    results: list[ArmResult] = []
    xt, xv = prepare_features(x_train, x_val, whiten=whiten)
    real_dim = xt.shape[1]  # whitening reduces dim; keep arms matched to it

    _, acc = euclidean_lr(xt, y_train, xv, y_val, seed=seed, prewhitened=True)
    results.append(ArmResult("euclidean_lr", acc, float("nan"), 0.0))

    def mk(**kw):
        base = dict(in_dim=real_dim, n_classes=n_classes, proj_dim=proj_dim,
                    seed=seed, epochs=epochs)
        base.update(kw)
        return ProbeConfig(**base)

    for name, cfg in [
        ("flat_on_transform", mk(curvature=1.0, use_manifold=False)),
        ("curvature_zero", mk(curvature=0.0)),
        ("hyperbolic", mk(curvature=curvature, use_manifold=True)),
    ]:
        _, res = fit_probe(xt, y_train, xv, y_val, cfg)
        results.append(ArmResult(name, res.val_acc, res.macro_f1, res.curvature))
    return results
