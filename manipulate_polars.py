from functools import reduce
from datetime import timedelta
import polars as pl
from windows import get_date_fields, not_finite_floats_to_none

regex_year = [r"(189[0-9]|19[0-9][0-9]|20[0-2][0-9])", pl.UInt16]

def are_keys(frame, keys_):
    frame_keys = frame.select(keys_)
    duplicates = frame_keys.filter(frame_keys.is_duplicated())
    n_nulls = frame_keys.null_count()
    n_nulls_sum = n_nulls.sum_horizontal().item()
    return duplicates.is_empty() and n_nulls_sum == 0, (duplicates, n_nulls)

def get_col_unique(frame, col):
    uniques = frame.get_column(col).unique()
    bol = uniques.shape == (1,) and uniques.null_count() == 0
    return None if uniques.is_empty() else (uniques[0], bol)

# def mask(frame, keys_vals):
#     anti = frame.join(keys_vals, on=keys, how='anti')
#     return frame.select(keys).join(anti, on=keys, how='left')

def columns_membership(columns, old, new):
    return (
        all(column in columns for column in old)
        and not any(column in columns for column in new)
    )

def join_many(frames, **kwargs):
    return reduce(lambda l, r: l.join(r, **kwargs), frames) if frames else None

def get_prod_uniques(frame, cols):
    uniques = [frame.select(pl.col(col).unique()) for col in cols]
    return join_many(uniques, how='cross')

def look_back(frame, key_time, keys_other, window):
    return (
        frame
        .groupby_rolling(key_time, keys_other, period=f"{window}d")
        .agg(pl.exclude([key_time, *keys_other]))
        .filter(pl.all(pl.exclude([key_time, *keys_other]).arr.lengths() >= window))
        .sort([key_time, *keys_other])
    )

def cast_to_date(frame, key_date, offset=timedelta(days=0)):
    date_fields_uniques = frame.select(get_date_fields(key_date)).unique()
    assert len(date_fields_uniques) == 1, date_fields_uniques
    return frame.with_columns((pl.col(key_date) + offset).cast(pl.Date))

# def two_columns_to_dict(records):
#     return dict(records.iter_rows())

def clean_concat(lst_dfs, **kwargs):
    data = None
    lst_dfs = [df for df in lst_dfs if not (df is None or df.is_empty())]
    if lst_dfs:
        data = pl.concat(lst_dfs, **kwargs)
    return data

def add_col(pl_df, col):
    if col not in pl_df.columns:
        pl_df = pl_df.with_columns(pl.lit(None).alias(col).cast(pl.Utf8))
    return pl_df

def add_cols(pl_df, cols):
    return reduce(add_col, cols, pl_df)

def get_daily_calendar(min_date, max_date, exclude_weekends=False, key_time='date'):
    interval = timedelta(days=1)
    dates = (
        pl.date_range(min_date, max_date, interval=interval, eager=True)
        .to_frame(key_time)
    )
    if exclude_weekends:
        dates = dates.select(pl.col(key_time).dt.weekday().is_in([6, 7]).not_())
    return dates

def get_full_index(frame, key_time, key_symbol):
    frame_time = frame.get_column(key_time)
    time = get_daily_calendar(frame_time.min(), frame_time.max())
    symbols = frame.select(pl.col(key_symbol).unique())
    return time.join(symbols, how='cross').sort(key_time, key_symbol)

def get_full_index_from_folded(frame, key_time, key_symbol, key_time_min=None, key_time_max=None):
    if key_time_min is None:
        key_time_min = f'min_{key_time}'
    if key_time_max is None:
        key_time_max = f'max_{key_time}'
    min_time = frame.get_column(key_time_min).min()
    max_time = frame.get_column(key_time_max).max()
    time = get_daily_calendar(min_time, max_time)
    assert time.columns[0] == key_time
    symbols = frame.select(pl.col(key_symbol).unique())
    return time.join(symbols, how='cross').sort(key_time, key_symbol)

def fold(frame, key_time, keys_other):
    return (
        frame
        .group_by(keys_other)
        .agg([
            pl.col(key_time).min().name.prefix('min_'),
            pl.col(key_time).max().name.prefix('max_')
        ])
        .sort(keys_other)
    )

def get_full_index_per_symbol(folded, key_time, key_symbol, key_time_min=None, key_time_max=None):
    if key_time_min is None:
        key_time_min = f'min_{key_time}'
    if key_time_max is None:
        key_time_max = f'max_{key_time}'
    full_index = get_full_index_from_folded(
        folded, key_time, key_symbol, key_time_min, key_time_max
    )
    melted = (
        folded
        .with_columns(pl.col(key_time_max) + timedelta(days=1))
        .melt(id_vars=key_symbol, value_vars=[key_time_min, key_time_max], value_name=key_time)
        .with_columns(pl.col('variable') == key_time_min)
    )
    return (
        full_index
        .join(melted, on=(key_time, key_symbol), how='left')
        .with_columns(pl.col('variable').forward_fill().over(key_symbol))
        .filter(pl.col('variable'))
        .select(key_time, key_symbol)
    )

def get_consecutive_slices(fully_indexed, key_time, keys_other, col):
    shifts = (
        fully_indexed
        .with_columns(pl.col(col).is_null().alias('shifted_0'))
        .with_columns(pl.col(col).shift(-1).over(keys_other).is_null().alias('shifted_-1'))
        .with_columns(pl.col(col).shift(+1).over(keys_other).is_null().alias('shifted_+1'))
        .filter(
            (pl.col('shifted_0') != pl.col('shifted_-1'))
            | (pl.col('shifted_0') != pl.col('shifted_+1'))
        )
        .with_columns([
            pl.when(pl.col('shifted_0') | pl.col('shifted_-1'))
            .then(False).otherwise(True).alias('is_start'),
            pl.when(pl.col('shifted_0') | pl.col('shifted_+1'))
            .then(False).otherwise(True).alias('is_end'),
        ])
        .filter(pl.col('is_start') | pl.col('is_end'))
    )
    consecutive_slices = (
        shifts
        .with_columns(pl.col('is_start').cum_sum().over(keys_other).alias('slice'))
        .group_by(*keys_other, 'slice')
        .agg([
            pl.col(key_time).min().name.prefix('start_'),
            pl.col(key_time).max().name.prefix('end_'),
        ])
        .sort(*keys_other, 'slice')
        .select(*keys_other, f'start_{key_time}', f'end_{key_time}')
    )
    return consecutive_slices

def pivot_to_pd(frame, index, columns, values):
    return (
        frame
        .pivot(
            index=index,
            columns=columns,
            values=values,
            maintain_order=True,
            sort_columns=True
        )
        .to_pandas()
        .set_index(index)
    )

def membership_turnover(universe, key_time, key_symbol):
    return (
        universe
        .select(key_time, key_symbol)
        .group_by(key_time, maintain_order=True)
        .agg(pl.col(key_symbol).alias('shifted_0'))
        # .agg(pl.col(key_symbol).map_elements(set))
        .with_columns([
            pl.col('shifted_0').shift(+1).alias('shifted_+1'),
            # pl.col('shifted_0').shift(-1).alias('shifted_-1'),
        ])
        # .drop_nulls()  # BUG: works but then the rest bugs
        .with_columns([
            pl.col('shifted_0').list.set_difference(pl.col('shifted_+1')).alias('n_in'),
            pl.col('shifted_+1').list.set_difference(pl.col('shifted_0')).alias('n_out'),
        ])
        .with_columns(pl.exclude(key_time).list.len())
        .select(key_time, 'n_in', 'n_out')
    )

def have_nulls_included(frame, col_1, col_2):
    return frame.filter(pl.col(col_1).is_null() & pl.col(col_2).is_not_null()).is_empty()

def have_same_nulls(frame, col_1, col_2):
    return have_nulls_included(frame, col_1, col_2) and have_nulls_included(frame, col_2, col_1)

def split_train_valid_test(frame, key_time, prop_train, prop_valid):
    props_sum = prop_train + prop_valid
    assert props_sum <= 1.0
    cum_props = (
        frame
        .sort(key_time)
        .group_by(key_time, maintain_order=True)
        .len()
        .with_columns(pl.col('len').cum_sum())
        .with_columns(pl.col('len') / pl.col('len').max())
    )
    end_date_train_ = cum_props.filter(pl.col('len') <= prop_train).row(-1)[0]
    end_date_valid_ = cum_props.filter(pl.col('len') <= props_sum).row(-1)[0]
    return end_date_train_, end_date_valid_, cum_props.row(-1)[0]

def get_leverage_one(frame, key_time, col):
    leverage_one = pl.col(col) / pl.col(col).abs().sum().over(key_time)
    return frame.with_columns(leverage_one=leverage_one)

def get_mean_zero(frame, key_time, col):
    mean_zero = pl.col(col) - pl.col(col).mean().over(key_time)
    return frame.with_columns(mean_zero=mean_zero)

def standard_scale(frame, key_time, col):
    standard_scaler = (pl.col(col) - pl.col('mean')) / pl.col('std')
    return (
        frame
        .with_columns([
            pl.col(col).mean().over(key_time).alias('mean'),
            pl.col(col).std().over(key_time).alias('std'),
        ])
        .with_columns(standard=standard_scaler)
        .with_columns(not_finite_floats_to_none('standard'))
    )

def robust_scale(frame, key_time, col, quantile=0.25):
    robust_scaler = (pl.col(col) - pl.col('median')) / (pl.col('high') - pl.col('low'))
    return (
        frame
        .with_columns([
            pl.col(col).median().over(key_time).alias('median'),
            pl.col(col).quantile(quantile).over(key_time).alias('low'),
            pl.col(col).quantile(1 - quantile).over(key_time).alias('high'),
        ])
        .with_columns(robust=robust_scaler)
        .with_columns(not_finite_floats_to_none('robust'))
    )

def get_rank(col, key_time):
    return (
        pl.col(col)
        .rank(method='ordinal', descending=True)
        .over(key_time)
        .cast(pl.Float64)
    )

def rank_to_long(leverage_one=True):
    unif = (pl.col('rank') - 1.0) / (pl.col('rank_max') - 1.0)
    leverage = 0.5 * pl.col('rank_max') if leverage_one else 1.0
    return unif / leverage

def rank_to_long_short():
    unif = 2 * pl.col('rank') - pl.col('rank_max') - 1.0
    leverage = (pl.col('rank_max').cast(pl.UInt64) ** 2 // 2).cast(pl.Float64)
    return unif / leverage

def rank_scale(frame, key_time, col):
    return (
        frame
        .with_columns(rank=get_rank(col, key_time))
        .with_columns(rank_max=pl.col('rank').max().over(key_time))
        .with_columns(uniform=rank_to_long_short())
    )

def smooth_ts(weights, window_size, col, min_abs_weight=1e-5):
    return (
        weights
            .with_columns(
                smoothed=pl.col(col).fill_null(0.0)
                .rolling_mean(window_size=window_size).alias('smoothed')
            )
            .with_columns(
                pl.when(pl.col('smoothed').abs().ge(min_abs_weight))
                .then(pl.col('smoothed'))
                .otherwise(None)
            )
        )

def get_unique_date(frame, key_time):
    assert key_time in frame.columns
    unique_date = frame.select(key_time).unique()
    assert len(unique_date) == 1
    return unique_date.item()
