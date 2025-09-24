from __future__ import annotations

from typing import Any, Optional, Tuple

def _import_optional(name: str):
    try:
        return __import__(name, fromlist=["_"])
    except Exception:
        return None


cp = _import_optional("cupy")
cupyx = _import_optional("cupyx")
np = __import__("numpy")
sp = __import__("scipy.sparse", fromlist=["csr_matrix", "issparse"])  # type: ignore


def get_xp(device: str = "cpu"):
    device = {}
    if device == "gpu":
        device = {
            'xp' : cp,
            'xsc' : rsc,
            'xsp' : csp,
            'xen' : cen,
            'xdf' : cd
        }
        return device
    if device == "cpu":
        device = {
            'xp' : np,
            'xsc' : sc,
            'xsp' : sp,
            'xen' : sen,
            'xdf' : pd
        }
        return device
    else:
        raise ValueError(
            "Device should either be CPU or GPU."
        )

def np_as_dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)

