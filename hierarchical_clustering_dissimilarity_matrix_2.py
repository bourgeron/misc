from collections import defaultdict
from itertools import islice, combinations
from scipy.cluster.hierarchy import fcluster, leaves_list, optimal_leaf_ordering
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from fastcluster import linkage
from data import load_histories


# hierarchical clustering of a similarity matrix
# https://gmarti.gitlab.io/ml/2022/09/17/hierarchical-pca-crypto-clustering.html

def get_linkage(corr_pd_df):
    dissimilarities = 1. - corr_pd_df.to_numpy()
    condensed = dissimilarities[np.triu_indices(len(corr_pd_df), k=1)]
    return linkage(condensed, method='ward'), condensed

def sort_corr(corr_pd_df):
    links, condensed = get_linkage(corr_pd_df)
    perm = leaves_list(optimal_leaf_ordering(links, condensed))
    cols = corr_pd_df.columns
    corr_vals = corr_pd_df.to_numpy()
    sorted_corr_pd_df = pd.DataFrame(
        index=cols[perm], columns=cols[perm], data=corr_vals[perm, :][:, perm])
    return sorted_corr_pd_df

def cut_linkage(links, n_clusters):
    inds = fcluster(links, n_clusters, criterion='maxclust')
    clusters = defaultdict(list)
    for ind, n_cluster in enumerate(inds):
        clusters[n_cluster].append(ind)
    return sorted(clusters.values(), key=min)

def plot_clusters(sorted_corr_df, clusters):
    clusters_sizes = [len(clu) for clu in clusters]
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(sorted_corr_df)
    sizes = np.cumsum([0] + clusters_sizes)
    assert (dim := len(sorted_corr_df)) == sum(clusters_sizes)
    for left, right in zip(sizes, sizes[1:]):
        plt.axvline(x=left, ymin=left / dim, ymax=right / dim, color='r')
        plt.axvline(x=right, ymin=left / dim, ymax=right / dim, color='r')
        plt.axhline(y=left, xmin=left / dim, xmax=right / dim, color='r')
        plt.axhline(y=right, xmin=left / dim, xmax=right / dim, color='r')
    plt.show()
    print(slice_lst(list(sorted_corr), clusters_sizes))

def slice_lst(lst, sizes):
    ite = iter(lst)
    return [list(islice(ite, size)) for size in sizes]


# example of clustering wrt co-occurrences of discrete variables

def p_val_chi2_categ_indep(categs, col_0, col_1):
    p_val = chi2_contingency(pd.crosstab(
        categs[col_0], categs[col_1]))[1]
    return [[col_0, col_1, p_val], [col_1, col_0, p_val]]


def p_vals_chi2_categs_indep(categs):
    lst = [
        p_val_chi2_categ_indep(categs, col_0, col_1)
        for col_0, col_1 in combinations(categs.columns, r=2)
    ]
    p_vals = (
        pd.DataFrame([ele for sub in lst for ele in sub])
        .pivot(index=0, columns=1, values=2)
        .fillna(0))
    p_vals.index.name = None
    p_vals.columns.name = None
    return p_vals


if __name__ == "__main__":
    # 1st example
    keys = ['Date', 'Ticker']
    corr_df = (
        load_histories()
        .sort(keys)
        .select([pl.col(keys), pl.col('Close').pct_change().over(keys[1])])
        .to_pandas()
        .pivot(index=keys[0], columns=keys[1], values='Close')
        .corr(method='spearman')
    )
    sorted_corr = sort_corr(corr_df)
    lnkg = get_linkage(sorted_corr)[0]

    clusts = cut_linkage(lnkg, 6)
    plot_clusters(sorted_corr, clusts)

    clusts = cut_linkage(lnkg, 8)
    plot_clusters(sorted_corr, clusts)

    # 2nd example
    STR = 'abcdefghijklmnopqrstuvwxyz'
    splits = [STR[3 * n : 3 * n + 3] for n in range(len(STR) // 3)]
    categs_ = pd.DataFrame([
        np.random.choice(a=list(chars), size=100)  # TODO: set seed
        for chars in splits]).T
    p_vals_ = p_vals_chi2_categs_indep(categs_)

    sorted_corr = sort_corr(1. - p_vals_)
    lnkg, _ = get_linkage(sorted_corr)
    clusts = cut_linkage(lnkg, 3)
    plot_clusters(sorted_corr, clusts)
