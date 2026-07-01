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


def whiten_fit(x: np.ndarray, eps: float = 1e-6):
    """Fit ZCA whitening on ``x``; return (transform_fn, mean, W)."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0, keepdims=True)
    cov = np.atleast_2d(np.cov(x - mu, rowvar=False))
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, eps, None)
    w = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    def transform(z: np.ndarray) -> np.ndarray:
        return (np.asarray(z, dtype=np.float64) - mu) @ w

    return transform, mu, w


def euclidean_lr(x_train, y_train, x_val, y_val, seed: int = 0):
    """Flat logistic-regression probe (sklearn) on whitened features."""
    from sklearn.linear_model import LogisticRegression

    tf, _, _ = whiten_fit(x_train)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(tf(x_train), y_train)
    pred = clf.predict(tf(x_val))
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
) -> list[ArmResult]:
    """Train every arm on the same split and return their metrics."""
    results: list[ArmResult] = []

    _, acc = euclidean_lr(x_train, y_train, x_val, y_val, seed=seed)
    results.append(ArmResult("euclidean_lr", acc, float("nan"), 0.0))

    def mk(**kw):
        base = dict(in_dim=in_dim, n_classes=n_classes, proj_dim=proj_dim,
                    seed=seed, epochs=epochs)
        base.update(kw)
        return ProbeConfig(**base)

    for name, cfg in [
        ("flat_on_transform", mk(curvature=1.0, use_manifold=False)),
        ("curvature_zero", mk(curvature=0.0)),
        ("hyperbolic", mk(curvature=curvature, use_manifold=True)),
    ]:
        _, res = fit_probe(x_train, y_train, x_val, y_val, cfg)
        results.append(ArmResult(name, res.val_acc, res.macro_f1, res.curvature))
    return results
