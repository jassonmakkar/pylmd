import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import pdist, squareform, jaccard
from scipy.stats import spearmanr, entropy
from scipy.cluster.hierarchy import average
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.sparse.linalg import eigsh
from sklearn.cluster import DBSCAN

import dynamicTreeCut

def CalculateGeneDistance(dat: pd.DataFrame, method = 'pearson'):
    """
    Calculate Gene Pairwise Distance.

    Calculates the pairwise distance between genes in an expression matrix.

    Parameters
    ----------
    dat : pandas.DataFrame
        Expression data with genes as rows and cells as columns.
    method : str
        Metric for measuring gene-gene distance:

        - 'pearson': Pearson correlation distance (1 - correlation).
        - 'euclidean': Euclidean distance.
        - 'KL': Kullback-Leibler divergence.
        - 'jaccard': Jaccard distance.
        - 'spearman': Spearman correlation distance (1 - correlation).

    Returns
    -------
    pandas.DataFrame
        A square distance matrix (genes x genes).
    """

    # Normalize each row to sum to 1.
    normalized_array = normalize(dat.values, norm='l1', axis=1)
    rho_dat = pd.DataFrame(normalized_array, index=dat.index, columns=dat.columns)

    if method == "pearson":
        corr = np.corrcoef(rho_dat.values)
        dist = 1.0 - corr
    elif method == "euclidean":
        dist = pairwise_distances(rho_dat.values, metric="euclidean")
    elif method == "KL":
        pass
    elif method == "jaccard":
        dat_binary = (dat.values > 0)
        dist = pairwise_distances(dat_binary, metric="jaccard")
    elif method == "spearman":
        corr = rho_dat.T.corr(method="spearman").values
        dist = 1.0 - corr

    labels = dat.index
    return pd.DataFrame(dist, index=labels, columns=labels)


def ClusterGenes(dist, clustering_method = "average", 
                 min_gene = 10, deepSplit = 2,
                 return_tree = False, 
                 filtered = True, accu = 0.75):
    """
    This function partitions genes based on their pairwise distances using various clustering methods.

    Parameters
    ----------
    dist : pandas.DataFrame
        The gene-gene distance matrix.
    clustering_method : str
        The method for clustering the genes. Options are "average", "single", "complete", "median", or "centroid".
    min_gene : int
        The minimum number of genes each group should contain.
    deepSplit : int
        Depth parameter for dynamic cut.
    return_tree : bool
        If True, returns a gene partition tree; otherwise returns only the gene partition.
    filtered : bool
        If True, apply SML-style eigen filter within each cluster.
    accu : float
        The threshold for filtering out genes.
        
    Returns
    -------
    pandas.Series
        Categorical labels indexed by gene, or (Series, linkage_matrix) if return_tree.
    """

    if not isinstance(dist, pd.DataFrame):
        raise TypeError("dist must be a pandas.DataFrame.")
    if dist.shape[0] != dist.shape[1]:
        raise ValueError("dist must be square.")
    if not dist.index.equals(dist.columns):
        raise ValueError("dist index and columns must match and be in the same order.")
    if np.isnan(dist.values).any():
        raise ValueError("dist contains NaNs.")
    if not np.allclose(dist.values, dist.values.T, atol=1e-8, rtol=0):
        raise ValueError("dist must be symmetric.")
    if not np.allclose(np.diag(dist.values), 0.0, atol=1e-12):
        raise ValueError("dist diagonal must be zero.")

    if clustering_method in ["single", "complete", "average", 
                             "weighted", "centroid", "median"]:
        gene_hree = linkage(squareform(dist.values, checks=False), method=clustering_method)
        gene_partition = dynamicTreeCut.cutreeHybrid(
            link=gene_hree, 
            distM=dist.values, 
            deepSplit=deepSplit, 
            pamStage=True, 
            pamRespectsDendro=True,
            minClusterSize=min_gene)
        
        # Attach names
        gene_partition = pd.Series(gene_partition, index=dist.index, dtype="category")

        # Relabel modules by dendrogram order
        leaf_order = leaves_list(gene_hree)
        label_pos = {dist.index[i]: pos for pos, i in enumerate(leaf_order)}
        med_pos = gene_partition.groupby(gene_partition).apply(lambda idx: np.median([label_pos[g] for g in idx.index]))
        # Alternative: med_pos = (gene_partition.index.to_series().map(label_pos).groupby(gene_partition).median().sort_values())
        order = med_pos.sort_values().index.tolist()
        gene_partition = gene_partition.cat.reorder_categories(order)
        # Alternative: gene_partition = gene_partition.cat.reorder_categories(list(med_pos.index), ordered=True)
        gene_partition = gene_partition.cat.rename_categories({cat: i+1 for i, cat in enumerate(gene_partition.cat.categories)})

        # this is somewhat vibe-coded, warning
        if filtered:
            D = dist.values
            S = 1.0 - D # D must be scaled to [0,1]
            keep = []
            for cat in gene_partition.cat.categories:
                genes = gene_partition.index[gene_partition == cat]
                if len(genes) < min_gene:
                    continue
                idx = dist.index.get_indexer(genes)
                A = S[np.ix_(idx, idx)]
                # top eigenpair of symmetric A
                if A.shape[0] > 200:
                    vals, vecs = eigsh(A, k=1, which='LA')
                else:
                    vals, vecs = np.linalg.eigh(A)
                    vecs = vecs[:, -1:]
                    vals = vals[-1:]
                scores = np.sqrt(float(vals[-1])) * np.abs(vecs[:, -1])
                keep_genes = [genes[i] for i, s in enumerate(scores) if s > 2*accu - 1]
                if len(keep_genes) >= min_gene:
                    keep.extend(keep_genes)
            gene_partition = gene_partition.loc[keep]
            gene_partition = gene_partition.cat.remove_unused_categories()

        return (gene_partition, gene_hree) if return_tree else gene_partition

    elif clustering_method == "dbscan":
        # model = DBSCAN(metric="precomputed", eps=0.5)
        # labels = model.fit_predict(dist.values)
        # return pd.Series(labels, index=dist.index, dtype="category")
        pass
    elif clustering_method == "hdbscan":
        pass
    else:
        raise ValueError(f"Method not found: {clustering_method}")