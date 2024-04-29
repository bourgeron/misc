from datetime import timedelta
import polars as pl
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (9, 5)

def plot_pd_frame(pd_frame, *, with_mean=False, **kwargs):
    plt.figure()
    axis = pd_frame.plot(linewidth=2.0, **kwargs)
    axis.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    axis.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    axis.minorticks_on()

    if with_mean:
        for col, val in pd_frame.mean().items():
            axis.axhline(
                y=val, color='r', linestyle='--', linewidth=2.0,
                label=f'{col}_mean: {round(val, 3)}'
            )

    axis.legend()
    plt.tight_layout()
    plt.show()

def plot_ts(frame, key, *, with_mean=False, **kwargs):
    plot_pd_frame(frame.to_pandas().set_index(key), with_mean=with_mean, **kwargs)

def plot_desc(descs, **kwargs):
    cols = descs.columns
    cols_groups = {
        'count': cols[cols.isin(['count'])],
        'moments': cols[cols.isin(['mean', 'std'])],
        'quantiles': cols[~cols.isin(['count', 'mean', 'std'])],
    }
    for cols_group in cols_groups.values():
        if (lst_cols := list(cols_group)):
            plot_pd_frame(descs.loc[:, lst_cols], with_mean=False, **kwargs)

def plot_desc_pfo_matrix(weights, percentiles=None, **kwargs):
    plot_desc(weights.T.describe(percentiles).T, **kwargs)

def count_infrequent_points(frame, time_key, other_keys, delta=timedelta(days=1), agg_per='year'):
    return (
        frame
        .filter(pl.col(time_key).diff().over(other_keys) > delta)
        .with_columns(getattr(pl.col(time_key).dt, agg_per)())
        .groupby([time_key, *other_keys])
        .count()
        .sort([time_key, *other_keys])
        .pivot(index=time_key, columns=other_keys, values='count', aggregate_function=None)
    )

def get_most_infrequent_points(frame, time_key, other_keys):
    return (
        frame
        .with_columns(pl.col(time_key).diff().over(other_keys).suffix('_diff'))
        .filter(pl.col(time_key + '_diff') == pl.col(time_key + '_diff').max().over(other_keys))
        .sort(time_key + '_diff')
    )
