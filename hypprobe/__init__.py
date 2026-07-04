"""hypProbe package.

Thread-oversubscription guard (IMPORTANT on many-core machines like a DGX):
our geometry/probe math runs on TINY tensors (5-64 dims). By default PyTorch/BLAS
spin up one thread per core (128+ on a DGX), and for tiny ops the thread-herding
overhead dwarfs the compute -- a suite that runs in ~11 s on a laptop took 250+ s
on a 128-core box, and STAGE=rung0's bootstrap math would be far worse.

We therefore cap the BLAS/OMP thread env vars to 1 AT IMPORT (before numpy/torch
read them), unless the user has already set them -- so an explicit
`OMP_NUM_THREADS=8 ...` is still respected. This makes single runs fast; genuine
parallelism should come from running independent jobs, not from oversubscribed
BLAS on microscopic tensors.
"""

import os as _os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, "1")

# Also cap torch's intra-op threads if torch is imported later in-process.
try:  # torch may not be installed in every context (e.g. data-only tooling)
    import torch as _torch
    _torch.set_num_threads(1)
except Exception:
    pass
