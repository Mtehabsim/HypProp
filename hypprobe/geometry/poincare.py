"""Poincare ball model of hyperbolic space (pure PyTorch, no geoopt).

We implement the operations a hyperbolic linear probe needs:
  - exponential / logarithmic maps at the origin
  - Mobius addition
  - geodesic (Poincare) distance
  - distance to a gyroplane (used for hyperbolic MLR logits)

Everything is parameterised by curvature ``c > 0`` (the manifold has sectional
curvature ``-c``). As ``c -> 0`` every operation reduces to its Euclidean
counterpart, which is what lets us compare a hyperbolic probe against a flat one
on equal footing (see probes/hmlr.py and the curvature->0 control).

Numerics follow the standard hyperbolic-DL practice (Ganea et al. 2018; Guo et
al. 2022): clamp points strictly inside the ball, clamp arcosh/artanh arguments
away from their singularities, and prefer float32+ (bf16 blows up near the
boundary). Callers are expected to pass float32/float64 tensors.
"""

from __future__ import annotations

import torch

# Small margin keeping points strictly inside the open ball and away from the
# singularities of artanh/arcosh. Tuned for float32.
_EPS = 1e-5
_BOUNDARY_EPS = 1e-5
_MIN_DENOM = 1e-15


def _sqrt_c(c: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Return ``sqrt(c)`` as a tensor broadcastable against ``ref``."""
    if not torch.is_tensor(c):
        c = torch.tensor(float(c), dtype=ref.dtype, device=ref.device)
    return c.clamp_min(0.0).sqrt()


def project_to_ball(x: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """Project ``x`` to lie strictly inside the Poincare ball of curvature ``c``.

    Points must satisfy ``sqrt(c) * ||x|| < 1``. We rescale any point that lands
    on or outside the boundary back to radius ``(1 - eps)/sqrt(c)``. For ``c<=0``
    this is a no-op (Euclidean space has no boundary).
    """
    if (not torch.is_tensor(c)) and c <= 0:
        return x
    sqrt_c = _sqrt_c(c, x)
    if torch.is_tensor(sqrt_c) and float(sqrt_c) == 0.0:
        return x
    norm = x.norm(dim=-1, keepdim=True).clamp_min(_MIN_DENOM)
    max_norm = (1.0 - _BOUNDARY_EPS) / sqrt_c
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


def expmap0(v: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """Exponential map at the origin: tangent vector ``v`` -> point on the ball.

    ``expmap0_c(v) = tanh(sqrt(c)||v||) * v / (sqrt(c)||v||)``. As ``c->0`` this
    tends to the identity ``v``.
    """
    sqrt_c = _sqrt_c(c, v)
    if float(sqrt_c) == 0.0:
        return v
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(_MIN_DENOM)
    scaled = torch.tanh((sqrt_c * v_norm).clamp(-15.0, 15.0)) / (sqrt_c * v_norm)
    return project_to_ball(scaled * v, c)


def logmap0(y: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """Logarithmic map at the origin: point ``y`` on the ball -> tangent vector.

    Inverse of :func:`expmap0`. As ``c->0`` tends to the identity ``y``.
    """
    sqrt_c = _sqrt_c(c, y)
    if float(sqrt_c) == 0.0:
        return y
    y = project_to_ball(y, c)
    y_norm = y.norm(dim=-1, keepdim=True).clamp_min(_MIN_DENOM)
    arg = (sqrt_c * y_norm).clamp(-1.0 + _EPS, 1.0 - _EPS)
    return torch.atanh(arg) / (sqrt_c * y_norm) * y


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """Mobius addition ``x (+)_c y`` on the Poincare ball.

    Reduces to ordinary vector addition as ``c->0``.
    """
    if float(_sqrt_c(c, x)) == 0.0:
        return x + y
    c_t = c if torch.is_tensor(c) else torch.tensor(float(c), dtype=x.dtype, device=x.device)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c_t * xy + c_t * y2) * x + (1 - c_t * x2) * y
    denom = (1 + 2 * c_t * xy + c_t * c_t * x2 * y2).clamp_min(_MIN_DENOM)
    return project_to_ball(num / denom, c)


def dist(x: torch.Tensor, y: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """Geodesic (Poincare) distance between ``x`` and ``y``.

    ``d_c(x,y) = (2/sqrt(c)) * artanh(sqrt(c) * ||(-x) (+)_c y||)``. As ``c->0``
    this tends to ``2||x-y||`` -> we return the Euclidean distance ``||x-y||`` in
    that limit for a clean flat baseline.
    """
    sqrt_c = _sqrt_c(c, x)
    if float(sqrt_c) == 0.0:
        return (x - y).norm(dim=-1)
    diff = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1).clamp_min(_MIN_DENOM)
    arg = (sqrt_c * diff_norm).clamp(-1.0 + _EPS, 1.0 - _EPS)
    return 2.0 / sqrt_c * torch.atanh(arg)


def dist0(x: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """Geodesic distance from the origin to ``x`` (encodes hierarchy depth)."""
    sqrt_c = _sqrt_c(c, x)
    if float(sqrt_c) == 0.0:
        return x.norm(dim=-1)
    x_norm = x.norm(dim=-1).clamp_min(_MIN_DENOM)
    arg = (sqrt_c * x_norm).clamp(-1.0 + _EPS, 1.0 - _EPS)
    return 2.0 / sqrt_c * torch.atanh(arg)


def lambda_x(x: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """Conformal factor ``lambda^c_x = 2 / (1 - c||x||^2)``."""
    c_t = c if torch.is_tensor(c) else torch.tensor(float(c), dtype=x.dtype, device=x.device)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    return 2.0 / (1.0 - c_t * x2).clamp_min(_MIN_DENOM)


def gyroplane_distance(
    x: torch.Tensor,
    p: torch.Tensor,
    a: torch.Tensor,
    c: float | torch.Tensor,
) -> torch.Tensor:
    """Signed distance from points ``x`` to gyroplanes defined by ``(p, a)``.

    This is the quantity Ganea et al. (2018) use for the logits of hyperbolic
    multinomial logistic regression. Each class ``k`` has an on-ball offset
    ``p_k`` and a tangent normal ``a_k``; the (unnormalised) signed distance is

        d_k(x) = (2 / (sqrt(c) ||a_k||)) *
                 arcsinh( 2 sqrt(c) <(-p_k) (+)_c x, a_k>
                          / ((1 - c||(-p_k) (+)_c x||^2) ||a_k||) )

    As ``c->0`` this reduces to the Euclidean signed distance
    ``2 <x - p_k, a_k> / ||a_k||`` (up to the shared factor), i.e. an affine
    logit -- exactly ordinary logistic regression.

    Shapes: ``x`` is ``(N, d)``, ``p`` and ``a`` are ``(K, d)``. Returns
    ``(N, K)``.
    """
    n, d = x.shape
    k = p.shape[0]
    sqrt_c = _sqrt_c(c, x)
    a_norm = a.norm(dim=-1).clamp_min(_MIN_DENOM)  # (K,)

    if float(sqrt_c) == 0.0:
        # Euclidean limit: affine logit 2 <x - p, a> / ||a||.
        diff = x.unsqueeze(1) - p.unsqueeze(0)  # (N, K, d)
        inner = (diff * a.unsqueeze(0)).sum(dim=-1)  # (N, K)
        return 2.0 * inner / a_norm.unsqueeze(0)

    # mobius add of (-p_k) and x for every (n, k) pair.
    x_e = x.unsqueeze(1).expand(n, k, d)
    p_e = p.unsqueeze(0).expand(n, k, d)
    mp_x = mobius_add(-p_e, x_e, c)  # (N, K, d)
    mp_x2 = (mp_x * mp_x).sum(dim=-1)  # (N, K)
    inner = (mp_x * a.unsqueeze(0)).sum(dim=-1)  # (N, K)
    denom = ((1.0 - (sqrt_c ** 2) * mp_x2) * a_norm.unsqueeze(0)).clamp_min(_MIN_DENOM)
    arg = 2.0 * sqrt_c * inner / denom
    return 2.0 / sqrt_c * torch.asinh(arg)


def pairwise_dist(x: torch.Tensor, c: float | torch.Tensor) -> torch.Tensor:
    """Full ``(N, N)`` matrix of geodesic distances for the rows of ``x``."""
    n = x.shape[0]
    xi = x.unsqueeze(1).expand(n, n, x.shape[-1]).reshape(n * n, -1)
    xj = x.unsqueeze(0).expand(n, n, x.shape[-1]).reshape(n * n, -1)
    return dist(xi, xj, c).reshape(n, n)
