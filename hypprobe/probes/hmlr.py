"""Hyperbolic multinomial logistic regression (H-MLR) and its flat limit.

This is the project's probe primitive. It IS logistic regression, just written
for curved space: the flat decision hyperplanes ``<w_k, x> + b_k`` become
gyroplanes on the Poincare ball, and the logit is the signed gyroplane distance
(Ganea et al. 2018). Setting curvature ``c = 0`` recovers ordinary multinomial
logistic regression exactly -- that identity is what makes the hyperbolic vs
flat comparison fair (same model, same capacity, only the geometry differs).

Pipeline for one probe:
  raw features -> optional low-rank projection W (shared, learnable)
              -> feature-norm clipping (Guo et al. 2022, avoids vanishing grads)
              -> exp map into the ball at curvature c
              -> gyroplane-distance logits -> softmax cross-entropy.

All pure PyTorch; no geoopt. Curvature can be fixed or learned; when learned we
optimise ``log c`` for positivity. Optimisation is plain Adam on the tangent
parameters, which is the standard "Euclidean parameters, hyperbolic output"
setup and avoids a Riemannian optimiser dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..geometry import poincare


@dataclass
class ProbeConfig:
    in_dim: int
    n_classes: int
    proj_dim: int = 5            # low-rank probe dim (hyperbolic wins live low)
    curvature: float = 1.0       # c; set 0.0 for the flat baseline
    learn_curvature: bool = False
    clip_norm: float = 15.0      # max feature norm before exp map (MDR-style)
    weight_decay: float = 1e-3
    lr: float = 1e-2
    epochs: int = 200
    seed: int = 0
    device: str = "cpu"
    # If False, skip the exp map and classify in the tangent/Euclidean space
    # using the *same* projection W -- this is the "flat-on-same-transform" arm
    # that isolates curvature from the projection nonlinearity.
    use_manifold: bool = True


class HyperbolicMLR(nn.Module):
    """H-MLR head with a shared linear projection and gyroplane logits."""

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        self.proj = nn.Linear(cfg.in_dim, cfg.proj_dim, bias=True)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.randn(cfg.proj_dim, cfg.in_dim, generator=g) * (1.0 / cfg.in_dim ** 0.5)
            )
            self.proj.bias.zero_()
        # Gyroplane parameters: offset p_k (on-ball, via its tangent) and normal a_k.
        self.p_tangent = nn.Parameter(torch.zeros(cfg.n_classes, cfg.proj_dim))
        self.a = nn.Parameter(torch.randn(cfg.n_classes, cfg.proj_dim, generator=g) * 0.1)
        if cfg.learn_curvature and cfg.curvature > 0:
            self.log_c = nn.Parameter(torch.tensor(float(np.log(cfg.curvature))))
        else:
            self.register_buffer("log_c", torch.tensor(float(np.log(cfg.curvature))
                                                        if cfg.curvature > 0 else -1e30))

    @property
    def c(self) -> float | torch.Tensor:
        if self.cfg.curvature <= 0:
            return 0.0
        return torch.exp(self.log_c)

    def _clip(self, z: torch.Tensor) -> torch.Tensor:
        """Feature-norm clipping (Guo et al. 2022): rescale norms above clip_norm."""
        if self.cfg.clip_norm is None or self.cfg.clip_norm <= 0:
            return z
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        scale = torch.clamp(self.cfg.clip_norm / norm, max=1.0)
        return z * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self._clip(z)
        c = self.c
        if not self.cfg.use_manifold or self.cfg.curvature <= 0:
            # Flat limit: gyroplane_distance with c=0 gives the affine logit.
            return poincare.gyroplane_distance(z, self.p_tangent, self.a, 0.0)
        z_ball = poincare.expmap0(z, c)
        p_ball = poincare.expmap0(self.p_tangent, c)
        return poincare.gyroplane_distance(z_ball, p_ball, self.a, c)


@dataclass
class FitResult:
    train_acc: float
    val_acc: float
    macro_f1: float
    curvature: float
    codelength_bits: float = 0.0     # MDL (online codelength) if computed
    history: list[float] = field(default_factory=list)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for k in range(n_classes):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ProbeConfig,
) -> tuple[HyperbolicMLR, FitResult]:
    """Train an H-MLR (or its flat limit) and return the model + metrics."""
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    model = HyperbolicMLR(cfg).to(dev)

    xt = torch.as_tensor(x_train, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y_train, dtype=torch.long, device=dev)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=dev)
    yv = torch.as_tensor(y_val, dtype=torch.long, device=dev)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history: list[float] = []
    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        logits = model(xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        tr_pred = model(xt).argmax(dim=-1).cpu().numpy()
        va_logits = model(xv)
        va_pred = va_logits.argmax(dim=-1).cpu().numpy()
    train_acc = float(np.mean(tr_pred == y_train))
    val_acc = float(np.mean(va_pred == y_val))
    f1 = _macro_f1(y_val, va_pred, cfg.n_classes)
    c_val = float(model.c) if cfg.curvature > 0 else 0.0
    return model, FitResult(train_acc, val_acc, f1, c_val, history=history)
