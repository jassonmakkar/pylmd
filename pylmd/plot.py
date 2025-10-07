import numpy as np
import scipy.sparse as sp
import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re
import seaborn as sns

def visualize_network(
    W,
    adata,
    layout_key: str = "X_umap"
):
    """
    Plot a graph using:
      - affinity/adjacency matrix (sym, possibly sparse)
      - 2D visual space from adata.obsm[layout_key]
      
    """
    if type(W) != sp.csr_matrix:
        W = W.get()
    
    # --- matrix to dense for edge width calc
    A = W.toarray() if sp.issparse(W) else np.asarray(W)
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

def visualize_diffusion(
    adata,
    rho,
    P_ls,
    gene,
    reduc = 'X_umap'
):
    """
    Visualize diffusion of an initial state over dyadic times.
    """
    coord = np.asarray(adata.obsm[reduc])
    N = coord.shape[0]

    init = np.asarray(rho.loc[gene, adata.obs_names].values)
    # Make it a (1, N) row vector
    if init.ndim == 1:
        init = init[None, :]
    if init.shape[0] != 1 and init.shape[1] == 1:
        init = init.T
        
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

        fig.suptitle(gene, fontsize=18, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes

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