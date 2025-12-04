from collections import defaultdict
from functools import reduce
from itertools import combinations, islice, product
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from scipy.cluster.hierarchy import fcluster, leaves_list, optimal_leaf_ordering, linkage
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

keys_illiquid = ('ID_DAY', 'ID_FEATURE')
columns_illiquid = (*keys_illiquid, 'RET_FEATURE')
keys_liquid = ('ID_DAY', 'ID_TARGET')
columns_liquid = (*keys_liquid, 'RET_TARGET')

cast_id_feature = pl.col('ID_FEATURE').str.split('RET_').list.get(1).cast(pl.UInt16)

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

# data generic

def are_keys(frame, keys_):
    frame_keys = frame.select(keys_).sort(keys_)

    # duplicates = frame_keys.filter(frame_keys.is_duplicated())
    duplicates = (
        frame_keys
        .with_columns(count_temp=pl.lit(1))
        .with_columns(pl.col('count_temp').sum().over(keys_))
        .filter(pl.col('count_temp').gt(1))
        .select(keys_)
        .join(frame, on=keys_, how='left')
    )

    # n_nulls = frame_keys.null_count()
    n_nulls = (
        frame_keys
        .select(pl.col(keys_).is_null().sum())
    )

    if isinstance(frame, pl.LazyFrame):
        duplicates = duplicates.collect()
        n_nulls = n_nulls.collect()

    n_nulls_sum = n_nulls.sum_horizontal().item()
    return duplicates.is_empty() and n_nulls_sum == 0, (duplicates, n_nulls)

def join_many(frames, **kwargs):
    return reduce(lambda l, r: l.join(r, **kwargs), frames) if frames else None

def get_prod_uniques(frame, cols):
    uniques = [frame.select(pl.col(col).unique()) for col in cols]
    return join_many(uniques, how='cross')

def plot_ts(frame, **kwargs):
    frame.to_pandas().set_index(frame.columns[0]).plot(grid=True, **kwargs)

def pairs_corr(columns, **kwargs):
    return [
        pl.corr(*pair, **kwargs).alias(str(ind))
        for ind, pair in enumerate(combinations(columns, r=2))
    ]

def variances(columns):
    return [pl.var(col) for col in columns]

def pairs_cov(columns):
    return [
        pl.cov(*pair).alias(str(ind))
        for ind, pair in enumerate(combinations(columns, r=2))
    ]

def condensed_and_diag_to_sym_matrix(condensed, diag):
    dim  = (1 + (1 + 8 * len(condensed)) ** .5) / 2
    assert dim.is_integer()
    dim = int(dim)
    sym_matrix = np.zeros((dim, dim))
    np.fill_diagonal(sym_matrix, diag)
    inds = np.triu_indices(dim, k=1)
    sym_matrix[inds] = sym_matrix.T[inds] = condensed
    return sym_matrix

def cov_to_corr(cov):
    std_dev = np.diag(cov) ** 0.5
    return cov / (std_dev[:, None] * std_dev)

def square_matrix_to_polars_three_columns(square, columns, three_columns):
    return (
        pl.DataFrame(
            product(columns, repeat=2),
            schema={three_columns[0]: pl.UInt16, three_columns[1]: pl.UInt16}
        )
        .with_columns(pl.DataFrame(square.ravel(), schema={three_columns[2]: pl.Float64}))
    )

# data specific

def cast_features(features):
    schema = {
        'ID': pl.Int64,
        'ID_DAY': pl.Int64,
        'ID_TARGET': pl.UInt16,
    }
    columns = features.collect_schema().names()
    schema.update({col: pl.Float64 for col in columns if col not in schema})
    return features.cast(schema)

def concatenate_and_unstack(train_features_filled, train_targets):
    concatenated = pl.concat([
        train_targets.rename({'ID_TARGET': 'ID', 'RET_TARGET': 'RET'}),
        train_features_filled.rename({'ID_FEATURE': 'ID', 'RET_FEATURE': 'RET'})
    ])
    return (
        concatenated
        .sort('ID_DAY', 'ID')
        .pivot(index='ID_DAY', on='ID')
        .drop('ID_DAY')
        .fill_null(0.)
    )

def get_correlations(train_features_filled, train_targets):
    train_unstacked = concatenate_and_unstack(train_features_filled, train_targets)

    columns = train_unstacked.columns
    corr_condensed = train_unstacked.select(pairs_corr(columns, method='spearman'))
    corr_matrix = condensed_and_diag_to_sym_matrix(
        corr_condensed.to_numpy().squeeze(),
        1.
    )
    correlation = square_matrix_to_polars_three_columns(
        corr_matrix, columns, ['ID_TARGET', 'ID_FEATURE', 'CORR']
    )
    corr_pd_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)

    cov_condensed = train_unstacked.select(pairs_cov(columns))
    cov_matrix = condensed_and_diag_to_sym_matrix(
        cov_condensed.to_numpy().squeeze(),
        train_unstacked.select(variances(columns)).to_numpy().squeeze(),
    )
    covariance = square_matrix_to_polars_three_columns(
        cov_matrix, columns, ['ID_TARGET', 'ID_FEATURE', 'COV']
    )
    variance = (
        covariance
        .filter(pl.col('ID_TARGET') == pl.col('ID_FEATURE'))
        .rename({'COV': 'VAR'})
        .drop('ID_FEATURE')
    )

    covariance = (
        covariance
        .join(variance, on='ID_TARGET', how='left')
        .with_columns(BETA=pl.col('COV') / pl.col('VAR'))
    )
    return correlation.join(covariance, on=('ID_TARGET', 'ID_FEATURE')), corr_pd_df

def get_neighbors(clusters, ids_targets, ids_features):
    # TODO: remove, memory intensive
    prefix = 'CLASS_LEVEL_'
    stacked = (
        clusters
        .with_columns(pl.all().cast(pl.UInt16))
        .unpivot(
            index='ID',
            variable_name='LEVEL',
            value_name='CLUSTER',
        )
        .with_columns(pl.col('LEVEL').str.split(prefix).list[1].cast(pl.UInt16))
    )
    neighbors = (
        join_many([stacked.lazy(), stacked.lazy()], how='cross', suffix='_FEATURE')
        .rename({
            'ID': 'ID_TARGET',
            'LEVEL': 'LEVEL_TARGET',
            'CLUSTER': 'CLUSTER_TARGET',
        })
        .filter(
            pl.col('LEVEL_TARGET') == pl.col('LEVEL_FEATURE'),
            pl.col('CLUSTER_TARGET') == pl.col('CLUSTER_FEATURE'),
        )
        .rename({
            'LEVEL_TARGET': 'LEVEL',
            'CLUSTER_TARGET': 'CLUSTER',
        })
        .drop('LEVEL_FEATURE', 'CLUSTER_FEATURE')
        .collect()
        .join(ids_targets, on='ID_TARGET')
        .join(ids_features, on='ID_FEATURE')
        .sort('ID_TARGET', 'LEVEL')
    )
    assert ids_targets.join(neighbors, on='ID_TARGET', how='anti').is_empty()
    assert ids_features.join(neighbors, on='ID_FEATURE', how='anti').is_empty()
    return neighbors

def get_closest_neighbors(neighbors, ids_targets, min_n_features):
    neighborhoods_sizes = (
        neighbors
        .group_by('ID_TARGET', 'LEVEL')
        .len()
    )
    smallest_neighborhoods = (
        neighborhoods_sizes
        .filter(pl.col('len').ge(min_n_features))
        .sort(('ID_TARGET', 'len'), descending=(False, False))
        .group_by('ID_TARGET')
        .first()
    )
    to_fill = ids_targets.join(smallest_neighborhoods, on='ID_TARGET', how='anti')
    default = (
        neighborhoods_sizes
        .join(to_fill, on='ID_TARGET')
        .filter(pl.col('LEVEL') == neighbors.get_column('LEVEL').min())
    )
    smallest_neighborhoods = pl.concat([smallest_neighborhoods, default])
    assert smallest_neighborhoods.select('ID_TARGET').sort('ID_TARGET').equals(ids_targets)
    closest_neighbors = neighbors.join(smallest_neighborhoods, on=('ID_TARGET', 'LEVEL'))
    assert closest_neighbors.select('ID_TARGET').unique().sort('ID_TARGET').equals(ids_targets)
    assert is_hierarchy(closest_neighbors, 'ID_TARGET', 'LEVEL')
    return closest_neighbors

# ML

def train_test_split(frame, date_col, test_size):
    frame_filtered = (
        frame
        .sort(date_col)
        .with_columns(count=pl.lit(1))
        .with_columns(total_count=pl.col('count').cum_sum())
        .with_columns(date_count=pl.col('total_count').last().over(date_col))
        .filter(pl.col('date_count') >= int((1 - test_size) * len(frame)))
    )
    return frame_filtered.head(1).get_column(date_col)[0], len(frame_filtered) / len(frame)

def get_shuffled_folds(n_folds, *frames):
    days = frames[0].select('ID_DAY').unique()
    n_days = days.height
    parts_days = (
        days

        .with_columns(random=pl.lit(np.random.rand(n_days)))
        .sort('random')
        .drop('random')

        .with_columns(count=pl.lit(1))
        .with_columns(pl.col('count').cum_sum())
        .with_columns((pl.col('count') / (n_days / n_folds)).cast(pl.UInt8))
        .filter(pl.col('count').le(n_folds - 1))

        .partition_by('count', include_key=False)
    )
    return [[frame.join(part_days, on='ID_DAY') for frame in frames] for part_days in parts_days]

def compute_metric(prediction):
    correct_signs = prediction.filter(pl.col('RET_TARGET').sign() == pl.col('PREDICTION').sign())
    return (
        correct_signs.get_column('RET_TARGET').abs().sum()
        / prediction.get_column('RET_TARGET').abs().sum()
    )

def metric(y_true, y_pred):
    have_correct_signs = (np.sign(y_true) == np.sign(y_pred)).astype(int)
    return np.sum(have_correct_signs * np.abs(y_true)) / np.sum(np.abs(y_true))

scoring = make_scorer(metric, greater_is_better=True)

# not smooth, cf. SmoothMetric
class Metric(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        have_correct_signs = (torch.sign(y_pred) == torch.sign(y_true)).float()
        score = (have_correct_signs * y_true.abs()).sum() / y_true.abs().sum()
        return 1.0 - score

def get_score_fold(train_predict_score, folds, ind, *params, **extra_data):
    train_feats, train_tgts = zip(*[folds[idx] for idx in range(len(folds)) if idx != ind])
    score = train_predict_score(
        pl.concat(train_feats),
        pl.concat(train_tgts),
        *folds[ind],
        *params,
        **extra_data
    )
    return ind, *params, score

def cross_validate(param_grid, folds, train_predict_score, **extra_data):
    series = [pl.DataFrame(values, schema=[key]) for key, values in param_grid.items()]
    params = join_many(series, how='cross')

    scores = [
        get_score_fold(train_predict_score, folds, param[0], *param[1:], **extra_data)
        for param in tqdm(params.iter_rows(), total=len(params))
    ]
    return pl.DataFrame(scores, schema=[*params.columns, 'score'], orient='row')

def join_truth_and_score(preds, test_days, test_targets):
    prediction = (
        pl.concat([test_days, preds], how='horizontal')
        .unpivot(index='ID_DAY', variable_name='ID_TARGET', value_name='PREDICTION')
        .with_columns(pl.col('ID_TARGET').cast(pl.UInt16))
    )
    with_target = test_targets.join(prediction, on=('ID_DAY', 'ID_TARGET'))
    assert len(with_target) == len(test_targets)
    return compute_metric(with_target)

def fill_null(frame, cols):
    return frame.with_columns(
        pl.col(cols[2]).fill_null(pl.col(cols[2]).mean().over(cols[0]))
    )

def xs_standardize(frame, cols):
    return (
        frame
        .with_columns(
            pl.col(cols[2]).mean().over(cols[0]).alias('mean'),
            pl.col(cols[2]).std().over(cols[0]).alias('std'),
        )
        .with_columns((pl.col(cols[2]) - pl.col('mean')) / pl.col('std'))
        .drop('mean', 'std')
    )

def roll_mean(frame, cols, n_days):
    return (
        frame
        .rolling(cols[0], period=f'{n_days}i', group_by=cols[1])
        .agg(pl.col(cols[2]).mean())
        .sort(cols[:2])
    )

def roll_standardize(frame, cols, n_days):
    rolled = (
        frame
        .rolling(cols[0], period=f'{n_days}i', group_by=cols[1])
        .agg(pl.col(cols[2]).mean().alias('mean'), pl.col(cols[2]).std().alias('std'))
    )
    return (
        frame
        .join(rolled, on=cols[:2])
        .with_columns((pl.col(cols[2]) - pl.col('mean')) / pl.col('std'))
        .drop('mean', 'std')
        .sort(cols[:2])
    )

def demean(frame, cols, supplementary_data, level):
    expr = pl.col(cols[2]) - pl.col(cols[2]).mean().over(cols[0], f'CLASS_LEVEL_{level}')
    match level:
        case 0:  # xs_demean
            frame = (
                frame
                .with_columns(pl.col(cols[2]) - pl.col(cols[2]).mean().over(cols[0]))
            )
        case 1 | 2 | 3 | 4:
            frame = (
                frame
                .join(supplementary_data.rename({'ID': cols[1]}), on=cols[1])
                .with_columns(expr)
                .sort(cols[:2])
            )
    return frame

def preprocess_one(frame, cols, preprocessing, **kwargs):
    frame = fill_null(frame, cols)
    match preprocessing:
        case 'roll':
            frame = roll_mean(frame, cols, kwargs['n_days'])
        case 'demean':
            frame = demean(frame, cols, kwargs['supplementary_data'], kwargs['level'])
        case 'roll+demean':
            frame = roll_mean(frame, cols, kwargs['n_days'])
            frame = demean(frame, cols, kwargs['supplementary_data'], kwargs['level'])
        case 'demean+roll':
            frame = demean(frame, cols, kwargs['supplementary_data'], kwargs['level'])
            frame = roll_mean(frame, cols, kwargs['n_days'])
    return frame

def prepare_data_for_numpy(frame, cols, preprocessing=None, **kwargs):
    frame = preprocess_one(frame, cols, preprocessing, **kwargs)
    return (
        frame
        .sort(cols[:2])
        .pivot(index=cols[0], on=cols[1], values=cols[2], sort_columns=True)
        .with_columns(pl.exclude(cols[0]).fill_null(0.))
    )

def prepare_data_for_ml(
    train_features, train_targets, test_features, test_targets,
    preprocessing=None, **kwargs
):
    train_features_rect = prepare_data_for_numpy(
        train_features, columns_illiquid, preprocessing, **kwargs
    )
    train_targets_rect = prepare_data_for_numpy(train_targets, columns_liquid, None)
    assert train_features_rect.select('ID_DAY').equals(train_targets_rect.select('ID_DAY'))
    train_days = train_features_rect.select('ID_DAY')

    test_features_rect = prepare_data_for_numpy(
        test_features, columns_illiquid, preprocessing, **kwargs
    )
    test_targets_rect = prepare_data_for_numpy(test_targets, columns_liquid, None)
    assert test_features_rect.select('ID_DAY').equals(test_targets_rect.select('ID_DAY'))
    test_days = test_features_rect.select('ID_DAY')

    assert train_features_rect.columns == test_features_rect.columns
    features = train_features_rect.drop('ID_DAY').columns
    assert train_targets_rect.columns == test_targets_rect.columns
    targets = train_targets_rect.drop('ID_DAY').columns
    return {
        'train_features_rect': train_features_rect.drop('ID_DAY'),
        'train_targets_rect': train_targets_rect.drop('ID_DAY'),
        'train_days': train_days,
        'test_features_rect': test_features_rect.drop('ID_DAY'),
        'test_targets_rect': test_targets_rect.drop('ID_DAY'),
        'test_days': test_days,
        'features': features,
        'targets': targets,
    }

def plot_learning_curve(train_sizes, train_scores, test_scores):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color=colors[0],
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color=colors[1],
    )
    plt.plot(
        train_sizes,
        train_scores_mean,
        'o-',
        color=colors[0],
        label='Training score',
    )
    plt.plot(
        train_sizes,
        test_scores_mean,
        'o-',
        color=colors[1],
        label='Validation score',
    )
    plt.legend(loc='best')
    plt.show()

    diffs = train_scores_mean - test_scores_mean
    last = f' {test_scores_mean[-1].round(3)} + {diffs[-1].round(3)}'
    plt.grid()
    plt.plot(
        train_sizes,
        diffs,
        'o-',
        color=colors[2],
        label='Last validation score' + last,
    )
    plt.legend(loc='best')
    plt.show()

def fit_predict_sklearn_model(
    train_features,
    train_targets,
    test_features,
    test_targets,
    sklearn_model,
):
    data = prepare_data_for_ml(train_features, train_targets, test_features, test_targets)
    sklearn_model.fit(data['train_features_rect'].to_numpy(), data['train_targets_rect'].to_numpy())
    predictions = sklearn_model.predict(data['test_features_rect'].to_numpy())
    return sklearn_model, pl.DataFrame(predictions, schema=data['targets'])

def heatmap_part(part, continuous, values='mean_test_score'):
    assert len(continuous) == 2

    heatmap_data = (
        part
        .select(*continuous, values)
        .sort(continuous, descending=(True, False))
        .pivot(index=continuous[0], on=continuous[1], values=values)
        .to_pandas()
        .set_index(continuous[0])
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        cmap='viridis',
        annot=True,
        fmt='.3f',
        cbar=False,
    )

    x_labels = part.get_column(continuous[1]).unique().round(5).sort()
    y_labels = part.get_column(continuous[0]).unique().round(5).sort(descending=True)

    plt.xticks(ticks=np.arange(len(x_labels)) + 0.5, labels=x_labels, rotation=45)
    plt.yticks(ticks=np.arange(len(y_labels)) + 0.5, labels=y_labels, rotation=0)

    plt.title(values)
    plt.xlabel(continuous[1])
    plt.ylabel(continuous[0])

def get_support(coef, sparsity):
    assert 0 <= sparsity < 1
    ind = int(sparsity * coef.size)
    abs_coef = np.abs(coef)
    threshold = np.partition(abs_coef.flatten(), ind)[ind]
    support = np.abs(coef) >= threshold

    empty_rows = np.sum(support, axis=1) == 0
    max_inds = abs_coef.argmax(axis=1)
    support[np.arange(coef.shape[0])[empty_rows], max_inds[empty_rows]] = True
    return support, np.sum(~support) / coef.size

def get_connected_components_supports(bipartite):
    n_targets, n_features = bipartite.shape
    n_assets = n_features + n_targets
    adjacency = np.zeros((n_assets, n_assets), dtype=bool)
    adjacency[:n_targets, n_targets:] = bipartite
    adjacency[n_targets:, :n_targets] = bipartite.T

    graph = nx.from_numpy_array(adjacency)
    components = nx.connected_components(graph)
    return [
        (
            np.array([ind + n_targets in component for ind in range(n_features)], dtype=bool),
            np.array([ind in component for ind in range(n_targets)], dtype=bool),
        )
        for component in components
        if any(asset < n_targets for asset in component)  # i.e. component has at least one target
    ]

# hierarchical clustering

def is_hierarchy(frame, granular_key, key):
    return (
        frame
        .group_by(granular_key, key)
        .len()
        .drop(key)
        .sort(granular_key)
        .equals(
            frame.group_by(granular_key).len().sort(granular_key)
        )
    )

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

def slice_lst(lst, sizes):
    ite = iter(lst)
    return [list(islice(ite, size)) for size in sizes]

def get_clusters(sorted_corr_df, clusters):
    clusters_sizes = [len(clu) for clu in clusters]
    return slice_lst(list(sorted_corr_df), clusters_sizes)

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
    return slice_lst(list(sorted_corr_df), clusters_sizes)

def dfs(node, visited, cluster, adjacency):
    visited.add(node)
    cluster.append(node)
    for neighbor in adjacency[node]:
        if neighbor not in visited:
            dfs(neighbor, visited, cluster, adjacency)

def get_clusters_from_adjacency(adjacency):
    visited = set()
    clusters = []
    for node in adjacency:
        if node not in visited:
            cluster = []
            dfs(node, visited, cluster, adjacency)
            clusters.append(sorted(cluster))
    return clusters

def get_clusters_from_mapping(mapping):
    adjacency = defaultdict(set)
    for row in mapping.iter_rows(named=True):
        adjacency[row['ID_FEATURE']].add(row['ID_TARGET'])
        adjacency[row['ID_TARGET']].add(row['ID_FEATURE'])

    clusters = get_clusters_from_adjacency(adjacency)

    flatten_clusters = [
        (ind, ele)
        for ind, cluster in enumerate(clusters)
        for ele in cluster
    ]
    schema = {'CLUSTER': pl.UInt16, 'ID': pl.UInt16}
    return pl.DataFrame(flatten_clusters, schema=schema, orient='row')

# PyTorch

def normalize_batch(batch):
    return (
        (batch - batch.mean(dim=0, keepdim=True))
        / batch.std(dim=0, keepdim=True)
    )

def train(sklearn_model, features, targets, patience=10):
    best_loss = float('inf')
    counter_no_improvement = 0

    if sklearn_model.has_lr_scheduler:
        scheduler = ReduceLROnPlateau(sklearn_model.optimizer, 'min', patience=5, factor=0.5)

    for epoch in range(sklearn_model.max_n_epochs):
        sklearn_model.model.train()
        running_loss = 0.0
        for ind in range(0, len(features), sklearn_model.batch_size):
            x_batch = features[ind:ind + sklearn_model.batch_size]
            y_batch = targets[ind:ind + sklearn_model.batch_size]
            if sklearn_model.is_std_scaled:
                x_batch = normalize_batch(x_batch)
            loss = sklearn_model.criterion(sklearn_model.model(x_batch), y_batch)

            sklearn_model.optimizer.zero_grad()
            loss.backward()
            if sklearn_model.has_lr_scheduler:
                scheduler.step(loss)
            sklearn_model.optimizer.step()

            torch.nn.utils.clip_grad_norm_(
                sklearn_model.model.parameters(),
                max_norm=sklearn_model.max_norm,
            )

            running_loss += loss.item()

        avg_loss = running_loss / (len(features) // sklearn_model.batch_size)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{sklearn_model.max_n_epochs}]. Loss: {avg_loss:.4f}.')

        if avg_loss < best_loss:
            best_loss = avg_loss
            counter_no_improvement = 0
        else:
            counter_no_improvement += 1
            if counter_no_improvement >= patience:
                print(f'Early stopped at {epoch + 1}. Final loss: {avg_loss:.4f}.')
                break

class SklearnNN(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model,
        learning_rate=0.001,
        max_n_epochs=100,
        batch_size=32,
        is_std_scaled=True,
        has_lr_scheduler=False,
        max_norm=1e2,
    ):
        locs = {
            'model': model,
            'learning_rate': learning_rate,
            'max_n_epochs': max_n_epochs,
            'batch_size': batch_size,
            'is_std_scaled': is_std_scaled,
            'has_lr_scheduler': has_lr_scheduler,
            'max_norm': max_norm,
        }
        for key, value in locs.items():  # `locs` stands for `locals()`
            if key != 'self':  # useless here
                setattr(self, key, value)
        self.criterion = SmoothMetric()  # torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def fit(self, features, targets):
        train(
            self,
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )
        return self

    def predict(self, features):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.tensor(features, dtype=torch.float32))
        return predictions.numpy()

    def get_params(self, deep=True):
        attrs = [
            'model', 'learning_rate', 'max_n_epochs', 'batch_size',
            'is_std_scaled', 'has_lr_scheduler', 'max_norm',
        ]
        return {attr: getattr(self, attr) for attr in attrs}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.criterion = SmoothMetric()  # torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self

def gen_layer(input_dim, output_dim, dropout_prob=0.0):
    return [
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.BatchNorm1d(output_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout_prob),
    ]

class SmoothMetric(torch.nn.Module):
    def __init__(self, scale=25):
        super().__init__()
        self.scale = scale

    def forward(self, y_pred, y_true):
        have_correct_signs = 0.5 * (torch.tanh(self.scale * y_pred) * torch.sign(y_true) + 1)
        score = (have_correct_signs * y_true.abs()).sum() / y_true.abs().sum()
        return 1.0 - score
