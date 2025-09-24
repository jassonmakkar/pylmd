from __future__ import annotations
from .utils import get_xp, np_as_dense
from typing import Optional, Dict, Any

def build_network(X, n_neighbors):
    Xnp = X if isinstance(X, np.ndarray) else X.A if hasattr(X, "A") else np.asarray(X)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(Xnp)
    inds = nn.kneighbors(Xnp, return_distance=False)
    rows = np.repeat(np.arange(Xnp.shape[0]), n_neighbors)
    cols = inds[:, 1:].reshape(-1)
    data = np.ones(rows.shape[0], dtype=np.float32)
    W = sp.csr_matrix((data, (rows, cols)), shape=(Xnp.shape[0], Xnp.shape[0]))
    W = W.maximum(W.T)
    return W

def visualize_network(
    affinity_m,
    adata,
    layout_key: str = "X_umap"
):
    """
    Plot a graph using:
      - affinity/adjacency matrix (sym, possibly sparse)
      - 2D visual space from adata.obsm[layout_key]
      
    """
    if type(W) != sp.csr_matrix:
        affinity_m = affinity_m.get()
    
    # --- matrix to dense for edge width calc
    A = affinity_m.toarray() if sp.issparse(affinity_m) else np.asarray(affinity_m)
    n = A.shape[0]

    if len(adata.obs_names) != n:
        raise ValueError(
            "nodes_obs_names not provided and adata.obs_names length "
            "does not match affinity matrix size."
        )
    idx = np.arange(n)
    obs_names_ordered = np.array(adata.obs_names)

    # --- get layout (2D)
    if layout_key not in adata.obsm:
        # helpful fallback suggestions
        candidates = [k for k in adata.obsm_keys() if k.lower().startswith("x_")]
        raise KeyError(
            f"'{layout_key}' not found in adata.obsm. "
            f"Available embeddings: {candidates or 'none'}"
        )
    layout_all = adata.obsm[layout_key]
    if layout_all.shape[1] < 2:
        raise ValueError(f"{layout_key} must have at least 2 dimensions (got {layout_all.shape[1]}).")
    layout = np.asarray(layout_all[idx, :2])  # align to nodes

    # --- build graph
    G = nx.from_numpy_array(A)  # undirected
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)

    # --- positions dict for networkx
    pos = {i: (layout[i, 0], layout[i, 1]) for i in range(n)}

    # --- edge widths ~ weight (map to ~0.5–1 like the R plot)
    weights = np.array([G[u][v].get("weight", 1.0) for u, v in G.edges], dtype=float)
    if weights.size:
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            widths = 0.5 + 0.5 * (weights - w_min) / (w_max - w_min)
        else:
            widths = np.full_like(weights, 0.75)
    else:
        widths = []

    node_colors = "black"
    legend_handles = None
    sm = None

    # --- draw
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="grey", alpha=0.7, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=5, ax=ax)

    ax.set_axis_off()

    return fig, ax

def rowwise_normalize(dat, W):
    """
    Row-normalize a genes x cells matrix, restricted to cells in W (matching column order).
    
    """
    
    # figure out cell order from W
    n = W.shape[0]
    cell_names = None

    if isinstance(dat, pd.DataFrame):
        # subset columns to match W column names
        if cell_names is None:
            # assume dat already restricted/ordered externally
            X = dat.values
            gene_names = dat.index
            cell_names = dat.columns
        else:
            X = dat.loc[:, cell_names].values
            gene_names = dat.index
    else:
        X = xp.asarray(dat)
        gene_names = np.arange(X.shape[0])
        cell_names = np.arange(X.shape[1])

    # row sums
    row_sums = X.sum(axis=1)
    keep = row_sums != 0
    X = X[keep]
    row_sums = row_sums[keep][:, None]

    rho = X / row_sums
    
    return pd.DataFrame(rho, index=np.array(gene_names)[keep], columns=cell_names)

def doubly_stochastic(W, device, max_iter: int = 100, tol: float = 1e-6):
    """
    Convert a symmetric affinity matrix to a doubly stochastic matrix using the Sinkhorn-Knopp algorithm.
    
    """

    xp = get_xp(device)
    
    r = xp.ones((W.shape[0], 1), dtype=W.dtype)
    c = xp.ones((1, W.shape[1]), dtype=W.dtype)
    for _ in range(max_iter):
        r_old = r
        c_old = c
        r = 1.0 / (W @ c.T + 1e-12)
        c = 1.0 / (r.T @ W + 1e-12)
        if xp.max(xp.abs(r - r_old)) < tol and xp.max(xp.abs(c - c_old)) < tol:
            break
            
    P = (r * W) * c
    
    return P


def construct_diffusion_operators(
    W, 
    device,
    max_time    
):
    """
    Constructs a list of diffusion operators for a given symmetric affinity matrix of a graph.
    
    """
    
    xp = get_xp(device)
    xsp = get_xsp(device)
    
    print("Creating diffusion operators...")

    if hasattr(W, "toarray"):
        W_dense = W.toarray()
    else:
        W_dense = W

    W_dense = xp.asarray(W_dense)
    
    P = doubly_stochastic(W_dense, device)
    
    n = P.shape[0]
    
    if max_time < 1:
        raise ValueError("Incorrect diffusion time, no propagation (max_time must be >= 1)")
    
    # Initialize list with identity matrix (t=0)
    P_dict = {0: xsp.identity(n, format='csr')}
    
    # Current P matrix
    P_current = P.copy()
    t = 1
    max_steps = int(xp.floor(xp.log2(max_time)))
    
    # Convergence criterion: check if diagonal approaches n
    convergence_threshold = n
    
    while t <= max_steps:
        # Check convergence: if diagonal elements approach uniform distribution
        diag_check = xp.abs((n * xp.diag(P_current)) * n - n)
        max_deviation = xp.max(diag_check)
        
        print(f"Step {t}: max deviation from convergence = {max_deviation}")
        
        if max_deviation < convergence_threshold:
            print(f"Converged at step {t} (time = {2**(t-1)})")
            break
        
        # Compute P^(2^t) = P^(2^(t-1)) * P^(2^(t-1))
        P_current = P_current @ P_current
        
        # Store as sparse matrix
        P_dict[2**t] = xsp.csc_matrix(P_current)
        
        t += 1
    
    actual_max_time = 2**(t-1) if t > 1 else 0
    print(f"Max diffusion time: {actual_max_time}")
    print(f"Total operators created: {len(P_dict)}")
    
    return P_dict

def visualize_diffusion(
    coord,
    rho,
    gene_name,
    P_ls=None,
    W=None
):
    """
    Visualize diffusion of an initial state over dyadic times.
    """
    coord = np.asarray(coord)
    N = coord.shape[0]

    init = np.asarray(rho.loc[gene, adata.obs_names].values)
    # Make it a (1, N) row vector
    if init.ndim == 1:
        init = init[None, :]
    if init.shape[0] != 1 and init.shape[1] == 1:
        init = init.T

    if P_ls is None:
        raise ValueError("Provide P_ls.")
        
    times = np.quantile(list(P_ls.keys()), [0, 0.33, 0.67, 1], method='higher')

    # --- compute multi_state for each time ---
    states = []
    max_vals = []
    for t in times:
        P = P_ls[t]
        P = P.get() if hasattr(P, "get") else P
        P = P if sp.issparse(P) else np.asarray(P)
        # init is (1,N); want (N, ) result transposed to (N,)
        state = (init @ P).ravel()
        states.append(state)
        max_vals.append(np.max(state) if state.size else 0.0)

    multi_state = np.column_stack(states)  # shape (N, T)
    # normalize each column to [0,1]
    col_max = np.array(max_vals)
    multi_state_norm = multi_state / col_max

    cmap = LinearSegmentedColormap.from_list("gene_grad", ["lightgrey", "blue"])

    # --- plot panels in one row ---
    n_panels = len(times)
    fig_w = 4 * n_panels
    fig_h = 4
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    # helper to build title like R: T=0 or T=2^{k}
    def _panel_title(t):
        if t == 0:
            return r"$T = 0$"
        k = int(np.log2(t))
        return rf"$T = 2^{{{k}}}$"

    for j, ax in enumerate(axes):
        vals = multi_state_norm[:, j]
        # order points so high values on top
        order = np.argsort(vals)
        x = coord[order, 0]
        y = coord[order, 1]
        c = vals[order]
        
        sc = ax.scatter(x, y, c=c, s=5, cmap=cmap, vmin=0.0, vmax=1.0, linewidths=0)
        ax.set_title(_panel_title(times[j]), fontsize=14, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_frame_on(False)
        
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Density", fontsize=10)
        cb.set_ticks([0.0, 1.0])

    if gene_name:
        fig.suptitle(gene_name, fontsize=18, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes

def calculate_score_profile(
    W,
    rho,
    P_ls,
    device
):
    
    xp = get_xp(device)
    xen = get_xen(device)
    xdf = get_xdf(device)
    
    Wn = W.shape[0]

    gene_names = rho.index.copy()
    X = rho.values
    X = xp.asarray(X)
    
    if Wn != X.shape[1] and Wn == X.shape[0]:
        X = X.T

    if Wn != X.shape[1]:
        raise ValueError("Check dims!")

    final_state = xp.full(Wn, 1.0 / Wn, dtype=float)

    # ----- compute per-time scores and cbind -----
    per_time_frames = []
    for P in P_ls:
              
        t = xen(X, (X @ P_ls[P]), axis=1)
        df_t = xdf.DataFrame(t, index=xp.arange(X.shape[0]))

        df_t.columns = [f"{col}_{P}" for col in df_t.columns]
        per_time_frames.append(df_t)

        print(f"Completed scoring for time {P}")

    score_df = xdf.concat(per_time_frames, axis=1)
    score_df.index = gene_names

    if device == 'gpu':
        X, final_state = xp.broadcast_arrays(X, final_state) # array broadcast not internally managed in cupy
    
    max_score = xen(X, final_state, axis=1)
    max_df = xdf.DataFrame(max_score, index=xp.arange(X.shape[0]))
    max_df.index = gene_names
    
    score_df = xdf.concat([score_df, max_df], axis=1)

    if device == 'gpu':
        score_df = score_df.to_pandas()

    return score_df

def obtain_lmds(score_df):
    """
    Compute LMDS (cumulative normalized diffusion KL score).
    """

    cols = score_df.columns.astype(str)
    score_cols = [c for c in cols if c.startswith("0_")]
    
    denom = score_df[0].replace(0, np.nan)
    norm_scores = score_df.copy()
    for col in score_df.columns:
        norm_scores[col] = score_df[col] / denom
    #norm_scores = (score_df).div(denom, axis=1) #not currently supported
    cum_score = norm_scores[score_cols].sum(axis=1)

    cum_score = cum_score.fillna(0.0)
    cum_score.name = "LMDS"
    
    norm_scores = norm_scores.drop(0, axis=1)
    
    return norm_scores, cum_score

def visualize_score_pattern(
    score_df,
    genes,
    figsize=(10, 6)
):

    # Validate genes
    genes = list(genes)
    missing = [g for g in genes if g not in score_df.index]
    if missing:
        raise ValueError(f"Genes not found: {missing[:3]}...")

    # Subset to genes of interest
    score_df = score_df.loc[genes].copy()
    
    # Find score columns (pattern: profile_time)
    score_cols = []
    pat = re.compile(r"^(.*)_(\d+)$")
    for col in score_df.columns:
        if pat.match(col):
            score_cols.append(col)
    
    # Melt to long format for plotting
    score_df['gene'] = score_df.index
    df = score_df.melt(id_vars=['gene'], value_vars=score_cols, 
                       var_name='time_col', value_name='score')
    
    # Extract time from column names
    df['time'] = df['time_col'].str.extract(r'_(\d+)$')[0].astype(int)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each gene as a separate line
    for gene in genes:
        gene_data = df[df['gene'] == gene].sort_values('time')
        ax.plot(gene_data['time'], gene_data['score'], 
                marker='o', label=gene, linewidth=2)

    ax.set_xscale('symlog', base=2)
    
    # Styling
    ax.set_xlabel('Time')
    ax.set_ylabel("Normalized Diffusion KL Score")
    ax.set_title('Gene Scores Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def find_knee(lmd_scores):
    
    v_sorted = np.sort(lmd_scores)
    
    x = np.arange(1, v_sorted.size + 1)
    x1, y1 = x[0], v_sorted[0]
    x2, y2 = x[-1], v_sorted[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = np.sqrt(dx * dx + dy * dy) + 1e-12
    A, B, C = dy, -dx, dx * y1 - dy * x1
    dist = np.abs(A * x + B * v_sorted + C) / norm
    knee = int(np.argmax(dist))

    return knee

def pyLMD(path, max_time = 2**20, device = 'cpu'):
    
    if device == 'gpu':
        import cupy as cp
        import cudf
        from cupyx.scipy import sparse as csp
        from cupyx.scipy.stats import entropy as cen
        import cudf as cd
        import rapids_singlecell as rsc
    elif device == "cpu":
        print('WARNING: Only running on CPU, please ensure this is intended.')
    else:
        raise ValueError("Device should either be CPU or GPU.")

    xsc = get_xsc(device)

    adata = sc.read_10x_h5(path)
    adata.var_names_make_unique()

    if device == 'gpu':
        rsc.get.anndata_to_GPU(adata)
    
    xsc.pp.filter_cells(adata, min_genes=100)
    xsc.pp.filter_genes(adata, min_cells=3)
    
    raw_adata = adata.copy()
    
    if device == 'gpu':      
        dat = pd.DataFrame(
            raw_adata.X.get().T.toarray() if not isinstance(raw_adata.X, np.ndarray) else raw_adata.X.T,
            index=adata.var_names,       # genes
            columns=adata.obs_names      # cells
        )
    else:
        dat = pd.DataFrame(
            raw_adata.X.T.toarray() if not isinstance(raw_adata.X, np.ndarray) else raw_adata.X.T,
            index=adata.var_names,       # genes
            columns=adata.obs_names      # cells
        )
    
    xsc.pp.normalize_total(adata)
    xsc.pp.log1p(adata)
    xsc.pp.highly_variable_genes(adata, n_top_genes=2000)
    xsc.pp.scale(adata,max_value=10)
    xsc.tl.pca(adata)
    if device == 'gpu':
        rsc.get.anndata_to_CPU(adata)
    xsc.pp.neighbors(adata)
    xsc.tl.umap(adata)
    
    cell_medians = np.median(dat, axis=0)
    mask = dat > cell_medians
    
    gene_detected_count = mask.sum(axis=1)
    
    selected_genes = (gene_detected_count >= 10) & (gene_detected_count <= dat.shape[1] * 0.3)
    
    adata = adata[:, selected_genes].copy()
    raw_adata = raw_adata[:, selected_genes].copy()
    dat = dat[selected_genes]
    
    feature_space = adata.obsm['X_pca'][:, :20]
    
    W = build_network(feature_space, n_neighbors=5)
    
    rho = rowwise_normalize(dat, W)
    
    P_ls = construct_diffusion_operators(W, device, max_time=max_time)
    
    score_df = calculate_score_profile(
        W = W,
        rho = rho,
        P_ls = P_ls,
        device = device
    )
    
    norm_scores, lmd_scores = obtain_lmds(score_df)
    
    sort_score = lmd_scores.sort_values()
    
    knee = find_knee(lmd_scores)
    
    LMDs = list(lmd_scores.sort_values()[:knee].index)

    return LMDs


#