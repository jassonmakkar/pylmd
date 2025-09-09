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


def get_xp(device: str = "auto"):
    if device == "gpu" and cp is not None:
        return cp
    if device == "cpu":
        return np
    return cp if cp is not None else np


def asxp(array, xp):
    if xp is np:
        if hasattr(array, "get"):
            return array.get()
        return np.asarray(array)
    if hasattr(array, "toarray") and not hasattr(array, "get"):
        return xp.asarray(array.toarray())
    return xp.asarray(array)


def to_sparse_csr(mat, xp):
    if xp is np:
        if sp.issparse(mat):
            return mat.tocsr()
        return sp.csr_matrix(mat)
    cx = cupyx.scipy.sparse if cupyx is not None else None
    if cx is None:
        raise RuntimeError("cupyx not available for sparse operations on GPU")
    if hasattr(mat, "tocsr") and isinstance(mat, cx.spmatrix):  # type: ignore
        return mat.tocsr()
    if sp.issparse(mat):
        return cx.csr_matrix((cp.asarray(mat.data), mat.indices, mat.indptr), shape=mat.shape)
    return cx.csr_matrix(xp.asarray(mat))


def rowwise_normalize(expr, axis=1, xp=None, eps=1e-12):
    xp = xp or np
    sums = expr.sum(axis=axis, keepdims=True)
    if hasattr(sums, "toarray"):
        sums = sums.toarray()
    sums = sums + eps
    return expr / sums


def sinkhorn_bistochastic(W, max_iter=50, tol=1e-6, xp=None):
    xp = xp or np
    if hasattr(W, "toarray"):
        Wd = W.toarray()
    else:
        Wd = W
    Wd = Wd.astype(Wd.dtype, copy=False)
    r = xp.ones((Wd.shape[0], 1), dtype=Wd.dtype)
    c = xp.ones((1, Wd.shape[1]), dtype=Wd.dtype)
    for _ in range(max_iter):
        r_old = r
        c_old = c
        r = 1.0 / (Wd @ c.T + 1e-12)
        c = 1.0 / (r.T @ Wd + 1e-12)
        if xp.max(xp.abs(r - r_old)) < tol and xp.max(xp.abs(c - c_old)) < tol:
            break
    P = (r * Wd) * c
    return P


def kl_divergence_rows(P, Q, xp=None, eps=1e-12):
    xp = xp or np
    P = xp.asarray(P)
    Q = xp.asarray(Q)
    P = xp.clip(P, eps, None)
    Q = xp.clip(Q, eps, None)
    return xp.sum(P * xp.log(P / Q), axis=1)


def matmul(A, B, xp=None):
    xp = xp or np
    return xp.matmul(A, B)


def knee_point(values, xp=None):
    xp = xp or np
    v = xp.asarray(values)
    v_sorted = xp.sort(v)
    x = xp.arange(1, v_sorted.size + 1)
    x1, y1 = x[0], v_sorted[0]
    x2, y2 = x[-1], v_sorted[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = xp.sqrt(dx * dx + dy * dy) + 1e-12
    A, B, C = dy, -dx, dx * y1 - dy * x1
    dist = xp.abs(A * x + B * v_sorted + C) / norm
    return int(xp.argmax(dist))

