from __future__ import annotations

from typing import Optional, Dict, Any

def get_xp(device: str = "auto"):
    if device == "gpu" and cp is not None:
        return cp
    if device == "cpu":
        return np
    return cp if cp is not None else np

def build_connectivities(adata, n_neighbors: int, use_rep: Optional[str], device: str):
    if getattr(adata, "obsp", None) is not None and "connectivities" in adata.obsp:
        return adata.obsp["connectivities"]
    
    # Build kNN graph
    X = adata.obsm[use_rep] if use_rep is not None and use_rep in adata.obsm else adata.X
    try:
        from cuml.neighbors import NearestNeighbors  # type: ignore
        use_gpu = device == "gpu"
    except Exception:
        use_gpu = False
    if use_gpu:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix  # type: ignore
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
        Xcp = cp.asarray(X)
        nn.fit(Xcp)
        inds = nn.kneighbors(Xcp, return_distance=False)
        
        # Build symmetric 0/1 adjacency
        rows = cp.repeat(cp.arange(Xcp.shape[0]), n_neighbors)
        cols = cp.asarray(inds[:, 1:]).reshape(-1)
        data = cp.ones(rows.shape[0], dtype=cp.float32)
        W = csr_matrix((data, (rows, cols)), shape=(Xcp.shape[0], Xcp.shape[0]))
        W = W.maximum(W.T)
        return W
    else:
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        from scipy.sparse import csr_matrix
        Xnp = X if isinstance(X, np.ndarray) else X.A if hasattr(X, "A") else np.asarray(X)
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(Xnp)
        inds = nn.kneighbors(Xnp, return_distance=False)
        rows = np.repeat(np.arange(Xnp.shape[0]), n_neighbors)
        cols = inds[:, 1:].reshape(-1)
        data = np.ones(rows.shape[0], dtype=np.float32)
        W = csr_matrix((data, (rows, cols)), shape=(Xnp.shape[0], Xnp.shape[0]))
        W = W.maximum(W.T)
    return W


def fast_calculate_score_profile(W, init_state, max_time_pow: int = 8, device: str = "auto"):
    xp = get_xp(device)
    
    # final bi-stationary distribution
    n = W.shape[0]
    final_state = xp.ones((n,), dtype=xp.float32) / n
    
    # doubly stochastic transition matrix P
    P = sinkhorn_bistochastic(_to_dense(W), xp=xp)

    # Dyadic diffusion times
    scores = {}
    curr_state = init_state
    P_pow = P

    # time 0
    score0 = kl_divergence_rows(curr_state, curr_state, xp=xp)  # zeros
    entropy = -kl_divergence_rows(curr_state, xp.ones_like(curr_state) / curr_state.shape[1], xp=xp)
    max_score0 = kl_divergence_rows(curr_state, final_state[None, :], xp=xp)
    scores[f"score0_0"] = score0

    for t in range(1, max_time_pow + 1):
        if t == 1:
            state_t = matmul(curr_state, P_pow, xp=xp)
        else:
            P_pow = matmul(P_pow, P_pow, xp=xp)
            state_t = matmul(curr_state, P_pow, xp=xp)
        s0 = kl_divergence_rows(curr_state, state_t, xp=xp)
        scores[f"score0_{2**t}"] = s0

    scores["entropy"] = entropy
    scores["max_score0"] = max_score0
    return scores


def obtain_lmds(score_profile: Dict[str, Any], correction: bool = False, device: str = "auto"):
    xp = get_xp(device)
    
    # Stack ordered score0_* keys
    score0_keys = sorted([k for k in score_profile.keys() if k.startswith("score0_")], key=lambda x: int(x.split("_")[-1]))
    score0 = xp.stack([score_profile[k] for k in score0_keys], axis=1)
    entropy = score_profile["entropy"]
    max_score0 = score_profile["max_score0"]
    if correction:
        df = score0 / (entropy[:, None] + 1e-12)
        lmds = xp.sum(df, axis=1)
    else:
        df = score0 / (max_score0[:, None] + 1e-12)
        lmds = xp.sum(df, axis=1)
    return lmds


def lmd(
    adata,
    n_neighbors: int = 30,
    use_rep: Optional[str] = "X_pca",
    max_time_pow: int = 8,
    min_cells: int = 5,
    correction: bool = False,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Compute LMD scores on an AnnData object using optional GPU acceleration.

    Returns a dict with 'score_profile' (dict of arrays), 'lmds' (array),
    'rank' (indices sorted by score), and 'knee_index'.
    """
    xp = get_xp(device)
    # Build connectivity graph
    W = _ensure_connectivities(adata, n_neighbors=n_neighbors, use_rep=use_rep, device=device)

    # Expression matrix
    X = adata.X 
    X = X.A if hasattr(X, "A") else X
    X = X.T
    X = X.astype("float32")
    
    gene_mask = (X > 0).sum(axis=1) >= min_cells
    X = X[gene_mask]

    rho = rowwise_normalize(X, axis=1, xp=xp)

    # Align columns to W ordering
    if rho.shape[1] != W.shape[0]:
        raise ValueError("Expression and graph dimension mismatch: genes x cells vs cells x cells")

    scores = fast_calculate_score_profile(W=W, init_state=rho, max_time_pow=max_time_pow, device=device)
    lmds = obtain_lmds(scores, correction=correction, device=device)

    # Ranking and knee point
    order = xp.argsort(lmds)
    kneedle = knee_point(lmds, xp=xp)

    gene_names = adata.var_names.to_numpy() if hasattr(adata, "var_names") else None
    if gene_names is not None:
        gene_names = gene_names[gene_mask]

    # Convert outputs to CPU numpy
    import numpy as np

    scores_cpu = {k: (v.get() if hasattr(v, "get") else np.asarray(v)) for k, v in scores.items()}
    lmds_cpu = lmds.get() if hasattr(lmds, "get") else np.asarray(lmds)
    order_cpu = order.get() if hasattr(order, "get") else np.asarray(order)

    return {
        "score_profile": scores_cpu,
        "lmds": lmds_cpu,
        "rank": order_cpu,
        "knee_index": int(kneedle),
        "genes": gene_names.tolist() if gene_names is not None else None,
        "gene_mask": gene_mask.tolist() if hasattr(gene_mask, "tolist") else gene_mask,
    }

