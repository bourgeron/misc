from collections import OrderedDict
from datetime import timedelta, date
from functools import reduce, partial
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from sys import stderr
import cvxpy as cp
from cvxpy import SolverError
import numpy as np
import pandas as pd
import polars as pl
# from sklearn.linear_model import LinearRegression
import ray
import matplotlib.pyplot as plt

keys = ['date', 'symbol']
init_date = date(2000, 1, 1)
end_date_train = date(2017, 4, 20)
EXIT_LENGTH = 21  # hard coded parameter in ccp.Stat_Arb.get_q
SOLVER = 'CLARABEL'
COST_SHORTING = 0.005 / 252

plt.rcParams['figure.figsize'] = (9, 5)

smoother_volumes = pl.col('volume').rolling_mean(window_size=5).over(keys[1])
turnover = pl.col('weight').diff().over(keys[1]).abs()

def are_keys(frame, keys_):
    frame_keys = frame.select(keys_)
    duplicates = frame_keys.filter(frame_keys.is_duplicated())
    n_nulls = frame_keys.null_count()
    n_nulls_sum = n_nulls.sum_horizontal().item()
    return duplicates.is_empty() and n_nulls_sum == 0, (duplicates, n_nulls)

def join_many(frames, **kwargs):
    return reduce(lambda l, r: l.join(r, **kwargs), frames) if frames else None

def get_prod_uniques(frame, cols):
    uniques = [frame.select(pl.col(col).unique()) for col in cols]
    return join_many(uniques, how='cross')

def get_daily_calendar(min_date, max_date, exclude_weekends=False, key_time=keys[0]):
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
    pd_frame = frame.to_pandas().set_index(key)
    plot_pd_frame(pd_frame, with_mean=with_mean, **kwargs)

def get_date_fields(key_date, fields=None):
    if fields is None:
        fields = ('hour', 'minute', 'second', 'millisecond', 'microsecond', 'nanosecond')
    return [getattr(pl.col(key_date).dt, field)().alias(field) for field in fields]

def cast_to_date(frame, key_date, offset=timedelta(days=0)):
    date_fields_uniques = frame.select(get_date_fields(key_date)).unique()
    assert len(date_fields_uniques) == 1, date_fields_uniques
    return frame.with_columns((pl.col(key_date) + offset).cast(pl.Date))

def loop_finite(fun, finite_iterable, processes, /, with_ray=True):
    # TODO: pass `processes` to `ray`
    if processes == 1:
        out = [fun(ele) for ele in finite_iterable]
    elif not with_ray:
        with Pool(processes) as pool:
            out = list(pool.imap_unordered(fun, finite_iterable))
    else:
        ray.init()
        out = ray.get([ray.remote(fun).remote(ele) for ele in finite_iterable])
        ray.shutdown()
    return out

def trade_one(stat_arb, formation, trading, t_max):
    formation_n_trading = pd.concat([formation, trading], axis=0)
    prices = formation_n_trading[stat_arb.asset_names] @ stat_arb.stocks
    if stat_arb.moving_mean:
        muu = prices.rolling(stat_arb.mu_memory).mean()
        muu = muu.iloc[-len(trading):]
    else:
        muu = stat_arb.mu
    return stat_arb.get_positions(trading, muu, T_max=t_max)

def trade_multi(stat_arbs, formation, trading, t_max):
    n_nulls = sum(1 for stat_arb in stat_arbs if stat_arb is None)
    positions = [
        trade_one(stat_arb, formation, trading, t_max)
        for stat_arb in stat_arbs
        if stat_arb is not None
    ]
    if positions:
        assigned = [position.assign(attempt=ind) for ind, position in enumerate(positions)]
    return n_nulls, pd.concat(assigned) if positions else None

def level_only(level):
    def inner(record):
        return record['level'].name == level
    return inner

def set_logger(logger, ordered_levels, logs_path):
    format_ = '{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}'
    logs_path = Path(logs_path)
    logs_path.mkdir(parents=True, exist_ok=True)
    logger.remove()
    for level in ordered_levels:
        logger.add(
            logs_path / f'log_{level.lower()}.log',
            filter=level_only(level),
            delay=True,
            format=format_
        )
    logger.add(logs_path / 'log_full.log', level=ordered_levels[0], delay=False, format=format_)
    logger.add(stderr, level=ordered_levels[-1], format=format_)

def plot_desc(descs, **plot_pars):
    cols = descs.columns
    cols_groups = {
        'count': cols[cols.isin(['count'])],
        'moments': cols[cols.isin(['mean', 'std'])],
        'quantiles': cols[~cols.isin(['count', 'mean', 'std'])],
    }
    for cols_group in cols_groups.values():
        if (lst_cols := list(cols_group)):
            plot_pd_frame(descs.loc[:, lst_cols], with_mean=False, **plot_pars)

def plot_desc_pfo_matrix(weights, percentiles=None, **plot_pars):
    plot_desc(weights.T.describe(percentiles).T, **plot_pars)

def columns_membership(columns, old, new):
    return (
        all(column in columns for column in old)
        and not any(column in columns for column in new)
    )

def prepend_index(frame, index_name='index', grp=None):
    assert columns_membership(frame.columns, [], [index_name])
    expr = pl.col(index_name).cumsum()
    if grp is not None:
        expr = expr.over(grp)
    return (
        frame
        .with_columns(pl.lit(1).alias(index_name).cast(pl.UInt64))
        .with_columns(expr - 1)
        .select(index_name, pl.exclude(index_name))
    )

def transaction_cost(smoothed_volume):
    params = {
        'log_coef': -0.175437,
        'const_coef': 4.169923,
        'mult_bias': 1.301015,
        'max': 90.0,
        'min': 0.1,
    }
    return (
        (params['const_coef'] + pl.col(smoothed_volume).log() * params['log_coef'])
        .exp()
        .mul(params['mult_bias'])
        .clip(params['min'], params['max'])
        .mul(1e-4)
    )

def market_impact_cost(smoothed_volume, volatility, coeff=1.0):
    return coeff * pl.col(volatility) * pl.col(smoothed_volume) ** -0.5

def load_and_prepare_market_data(
    *,
    path_market_data='prices_yf.parquet',
    start_date=init_date,
    volume_smoother=smoother_volumes,
    cost_transaction=transaction_cost('volume_smoothed'),
    cost_shorting=COST_SHORTING,
    cost_market_impact=market_impact_cost('volume_smoothed', 'volatility'),
):
    market_data = (
        pl.scan_parquet(path_market_data)
        .filter(pl.col(keys[0]) >= start_date)
    )

    dates = (
        market_data
        .select(pl.col(keys[0]).unique(maintain_order=True))
        .with_columns(pl.lit(1).alias(f"{keys[0]}_index"))
        .with_columns(pl.col(f"{keys[0]}_index").cum_sum() - 1)
    )

    n_days_listed = pl.col(f"{keys[0]}_index") - pl.col(f"{keys[0]}_index").min().over(keys[1]) + 1
    return (
        market_data
        .join(dates, on=keys[0], how='left')
        .with_columns([
            n_days_listed.alias('n_days_listed'),
            volume_smoother.alias('volume_smoothed'),
            pl.col('price').pct_change().std().over(keys[1]).alias('volatility')
        ])
        .with_columns(cost_shorting=cost_shorting)
        .with_columns([
            cost_transaction.alias('cost_transaction'),
            cost_market_impact.alias('cost_market_impact')
        ])
        .with_columns(not_finite_floats_to_none('cost_market_impact'))
    )

def split_market_data(market_data, formation_length, trading_length, n_stocks):
    sum_lengths = trading_length + formation_length
    market_data = (
        market_data
        .with_columns(
            batch=pl.col(f"{keys[0]}_index") // sum_lengths,
            is_formation=pl.col(f"{keys[0]}_index") % sum_lengths < formation_length,
        )
        .filter(pl.col('batch') != pl.col('batch').max())
    # last batch is removed because it is not complete with proba 1 - 1 / sum_lengths
    )

    is_first_formation_date = (
        pl.when(
            (pl.col(keys[0]) == pl.col(keys[0]).min().over('batch', 'is_formation'))
            & pl.col('is_formation'))
        .then(True)
        .otherwise(False)
    )
    volume_rank = pl.col('volume_smoothed').rank(method='ordinal', descending=True).over(keys[0])
    universe = (
        market_data
        .filter(is_first_formation_date)
        .filter(pl.col('n_days_listed') >= formation_length)
        .filter(volume_rank <= n_stocks)
        .select(*keys, pl.col('is_formation').alias('is_in_universe'))
    )
    return (
        market_data
        .join(universe, on=keys, how='left')
        .filter(pl.col('is_in_universe').forward_fill().over(keys[1], 'batch'))
        .select(*keys, 'price', 'volume', 'volume_smoothed', 'batch', 'is_formation')
    )

def get_universe(
    formation_length, trading_length, n_stocks,
    *, path_market_data='prices_yf.parquet', start_date=init_date, volume_smoother=smoother_volumes
):
    market_data = load_and_prepare_market_data(
        path_market_data=path_market_data,
        start_date=start_date,
        volume_smoother=volume_smoother,
        cost_transaction=transaction_cost('volume_smoothed'),
        cost_shorting=COST_SHORTING,
        cost_market_impact=market_impact_cost('volume_smoothed', 'volatility'),
    )
    universe = split_market_data(market_data, formation_length, trading_length, n_stocks)
    return universe

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

def get_splits(universe):
    splits = universe.partition_by('batch', 'is_formation', as_dict=True)
    batches = list(range(1, len(splits) // 2 + 1))
    assert len(splits) % 2 == 0
    assert sorted({batch for batch, _ in list(splits)}) == batches
    return [
        (
            pivot_to_pd(splits[batch, True], *keys, 'price'),
            pivot_to_pd(splits[batch, False], *keys, 'price')
        )
        for batch in batches
    ]

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

def get_universe_with_start_date(params):
    assert isinstance(params, dict)
    params_ = params.copy()
    params_names = ['formation_length', 'n_stocks', 'path', 'start_date', 't_max']
    assert sorted(params_) == params_names

    params_['trading_length'] = params_['t_max'] + EXIT_LENGTH
    del params_['t_max']

    path = params_.pop('path')
    universe = get_universe(**params_)
    return universe.collect().with_columns(pl.lit(path).alias('path'))

def read_data_one_run(path_run, path_index):
    path_run = Path(path_run)

    params = (
        pl.scan_parquet(Path(path_index))
        .filter(pl.col('path') == str(path_run))
        .collect()
    )
    assert len(params) == 1
    params = params.row(0, named=True)

    path_universe = path_run.parent / f'universe_{path_run.stem}.parquet'
    return params, pl.scan_parquet(path_run), pl.scan_parquet(path_universe)

def read_weights_one_hyperparameter(hyperparameter, path_index):
    paths = (
        pl.read_parquet(path_index)
        .join(pl.DataFrame(hyperparameter), on=list(hyperparameter))
        .get_column('path')
    )
    universes = []
    weights = []
    for path_run in paths:
        _, wghts, universe = read_data_one_run(path_run, path_index)
        # assert wghts.join(universe, on=keys, how='anti').is_empty()
        universes.append(universe)
        medianed = wghts.group_by(keys).agg(pl.col('weight').median())
        normalized = (
            universe
            .join(medianed, on=keys, how='inner')
            .with_columns(is_trade_first_day=(
                pl.col(keys[0])==pl.col(keys[0]).min().over('batch', 'is_formation')
            ))
            .with_columns(is_trade_first_day=(
                pl.when(pl.col('is_formation') > 0.5)
                .then(False)
                .otherwise(pl.col('is_trade_first_day'))
            ))
            .with_columns(leverage_first_day=pl.col('weight').abs().sum().over(keys[0]))
            .with_columns(leverage_first_day=(
                pl.when(pl.col('is_trade_first_day').not_())
                .then(None)
                .otherwise(pl.col('leverage_first_day'))
            ))
            .with_columns(pl.col('leverage_first_day').forward_fill().over('batch'))
            .with_columns(weight_normalized=pl.col('weight') / pl.col('leverage_first_day'))
            .select(*keys, pl.col('weight_normalized').alias('weight'))
            .sort(keys)
        )
        weights.append(normalized)
    universe = (
        pl.concat(universes)
        .select(keys)
        .unique()
        .sort(keys)
    )
    weights = (
        pl.concat(weights)
        .group_by(keys)
        .agg(pl.col('weight').median())
        .sort(keys)
    )
    return universe.join(weights, how='left', on=keys)

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

def not_finite_floats_to_none(col):
    return (
        pl.when(pl.col(col).is_nan() | pl.col(col).is_infinite())
        .then(None).otherwise(pl.col(col))
        .alias(col)
    )

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

def solve_optimization_problem(objective, constraints, *variables, logger=None):
    problem = cp.Problem(objective, constraints)
    value = None
    errored = True
    try:
        problem.solve(solver=SOLVER)
    except SolverError:
        pass
    else:
        if problem.status == cp.OPTIMAL:
            value = variables[0].value
            errored = False
    if errored:
        msg = f'Solver {SOLVER} failed: SolverError or not optimal status.'  # problem.status
        log(logger, msg, 'warning')
    return value, errored

def _project_lp_ball(vector, *, power=1, radius=1):
    variable = cp.Variable(len(vector))
    objective = cp.Minimize(cp.norm(vector - variable))
    constraints = [cp.norm(variable, power) <= radius, cp.sum(variable) == 0]
    return solve_optimization_problem(objective, constraints, variable)[0]

def _project_l1_sphere(vector, *, radius=1):
    variable = cp.Variable(len(vector))
    positive_part = cp.Variable(len(vector), nonneg=True)
    negative_part = cp.Variable(len(vector), nonneg=True)
    objective = cp.Minimize(cp.norm(vector - variable))
    constraints = [
        variable == positive_part - negative_part,
        cp.sum(positive_part) == 0.5 * radius,
        cp.sum(negative_part) == 0.5 * radius,
        cp.sum(variable) == 0
    ]
    return solve_optimization_problem(objective, constraints, variable)[0]

def extend_numpy_function(vector, function, *args, **kwargs):
    result = np.empty(len(vector))
    result.fill(np.nan)
    if not (are_nan := np.isnan(vector)).all():
        result[~are_nan] = function(vector[~are_nan], *args, **kwargs)
    return result

project_lp_ball = partial(extend_numpy_function, function=_project_lp_ball)
project_l1_sphere = partial(extend_numpy_function, function=_project_l1_sphere)

def extend_to_pl_cross_sectional(weights, date_key, col, *, function):
    # TODO: the signature could be improved, done like this to match the one of `rank_scale`
    unstacked = (
        weights
        .pivot(index=keys[1], columns=date_key, values=col)
    )
    applied = np.apply_along_axis(function, axis=0, arr=unstacked.drop(keys[1]).to_numpy())
    applied = pl.DataFrame(data=applied, schema=unstacked.columns[1:], nan_to_null=True)
    stacked = (
        pl.concat([unstacked.select(keys[1]), applied], how='horizontal')
        .melt(keys[1], variable_name=date_key, value_name='projected')
        .select(date_key,  keys[1], 'projected')
        .sort(date_key,  keys[1])
        .with_columns(pl.col(date_key).cast(pl.Date()))
    )
    return weights.join(stacked, on=keys, how='left')

project_l1_ball_pl = partial(extend_to_pl_cross_sectional, function=project_lp_ball)
project_l1_sphere_pl = partial(extend_to_pl_cross_sectional, function=project_l1_sphere)

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

def do_nothing(weights, *_):
    return weights.with_columns(nothing_done=pl.col('weight'))

def smooth_and_normalize(weights, params):
    outputs_columns = {
        'get_leverage_one': 'leverage_one',
        # 'get_mean_zero': 'mean_zero',
        # 'project_l1_ball_pl': 'projected',
        # 'standard_scale': 'standard',
        # 'robust_scale': 'robust',
        'rank_scale': 'uniform',
        'do_nothing': 'nothing_done',
    }
    weights_1 = (
        globals()[params['first_step']](weights, keys[0], 'weight').drop('weight')
        .rename({outputs_columns[params['first_step']]: 'weight'})
    )
    weights_2 = (
        smooth_ts(weights_1, params['window'], 'weight').drop('weight')
        .rename({'smoothed': 'weight'})
    )
    weights_3 = (
        globals()[params['third_step']](weights_2, keys[0], 'weight').drop('weight')
        .rename({outputs_columns[params['third_step']]: 'weight'})
    )
    # thanks to these last three preprocessing, a single weight on a day is set to (nan then) None
    weights_4 = (
        get_mean_zero(weights_3, keys[0], 'weight').drop('weight')
        .rename({'mean_zero': 'weight'})
    )
    weights_5 = (
        get_leverage_one(weights_4, keys[0], 'weight').drop('weight')
        .rename({'leverage_one': 'weight'})
    )
    return weights_5.select(*keys, not_finite_floats_to_none('weight'))

def get_index_loop(dol):
    index = [pl.DataFrame(vals, schema=[name]) for name, vals in dol.items()]
    return join_many(index, how='cross')

def read_data_one_preprocessing(path_preprocessing, path_index):
    path_preprocessing = Path(path_preprocessing)
    params = (
        pl.scan_parquet(Path(path_index))
        .filter(pl.col('path_preprocessing') == str(path_preprocessing))
        .collect()
    )
    assert len(params) == 1
    params = params.row(0, named=True)
    return params, pl.scan_parquet(params['path_preprocessing'])

def compute_metrics(weights, market_data):
    assert 'forward_return' in market_data.schema

    weights_n_market_data = market_data.join(weights, how='left', on=keys)

    ret = pl.col('weight') * pl.col('forward_return')
    underwater = (pl.col('cummax') - pl.col('cumsum')) / pl.col('cummax')
    metrics = [
        pl.col('turnover').mean().alias('mean_turnover'),  # independent of returns
        pl.col('return').mean().alias('raw_mean'),
        pl.col('return').std().alias('raw_std'),
        pl.col('underwater').max().alias('max_dd'),
        pl.col('underwater').mean().alias('mean_dd'),
    ]
    annualization = [
        (pl.col('raw_mean') / 1 * 1e4).alias('roe'),
        (pl.col('raw_mean') * 252).alias('mean'),
        (pl.col('raw_std') * 252 ** .5).alias('vol'),
        ((pl.col('raw_mean') * 252) / (pl.col('raw_std') * 252 ** .5)).alias('sharpe'),
        ((pl.col('raw_mean') * 252) / pl.col('mean_dd')).alias('calmar'),
    ]
    rot = pl.col('mean') / pl.col('mean_turnover')

    pnl = (
        weights_n_market_data
        .with_columns(pl.col('weight').fill_null(0.0))
        .with_columns([
            turnover.alias('turnover'),
            ret.alias('return')
        ])
        .group_by(keys[0], maintain_order=True)
        .agg([
            pl.col('turnover').sum(),
            pl.col('return').sum(),
            (pl.col('cost_transaction') * pl.col('turnover')).sum(),
            (pl.col('cost_shorting') * (-pl.col('weight').clip(-float('inf'), 0))).sum(),
        ])
        .with_columns([
            pl.col('return').cum_sum().alias('cumsum'),
            pl.col('cost_transaction').cum_sum(),
            pl.col('cost_shorting').cum_sum(),
        ])
        .with_columns(pl.col('cumsum') - pl.col('cost_transaction') - pl.col('cost_shorting') + 1.0)
        .with_columns(pl.col('cumsum').diff().alias('return'))
        .with_columns(cummax=pl.col('cumsum').cum_max())
        .with_columns(underwater=underwater)
    )

    metrics = (
        pnl
        .select(metrics)
        .with_columns(annualization)
        .with_columns(rot=rot)
        .select(pl.exclude('raw_mean', 'raw_std'))
    )
    return pnl, metrics

def log(logger, msg, level='info'):
    if logger is not None:
        getattr(logger, level)(msg)

def get_unique_date(frame, key_time):
    assert key_time in frame.columns
    unique_date = frame.select(key_time).unique()
    assert len(unique_date) == 1
    return unique_date.item()

def get_union_ytd_tdy(tdy_data, weights_ytd):
    tdy_data['weight_ytd'] = weights_ytd
    tdy_data = tdy_data[tdy_data['weight'].notna() | tdy_data['weight_ytd'].notna()]
    tdy_data.loc[:, 'weight'] = tdy_data['weight'].fillna(0.0)
    tdy_data.loc[:, 'weight_ytd'] = tdy_data['weight_ytd'].fillna(0.0)
    return tdy_data

def get_weights_missing_market_data(tdy, tdy_data, logger):
    market_data_tdy = tdy_data.drop(columns=['weight', 'weight_ytd'])
    symbols_wo_market_data_tdy = market_data_tdy[market_data_tdy.isna().any(axis=1)].index
    if not symbols_wo_market_data_tdy.empty and logger is not None:
        symbols = symbols_wo_market_data_tdy.tolist()
        policy = (
            f'On day {tdy}, the stocks {symbols} have missing market data.'
            ' The weights on these stocks will be set to 0.0'
            ' (without looking at the signal or constraints).'
        )
        log(logger, policy, 'debug')
    tdy_data_updated = tdy_data[~tdy_data.index.isin(symbols_wo_market_data_tdy)]
    weights = pd.Series(data=0.0, index=symbols_wo_market_data_tdy, name='weight')
    return tdy_data_updated, weights

def fetch_covariance(tdy, tdy_data_updated, covariances_path, build_objective, logger):
    weights = pd.Series()
    tdy_covariance = None
    if 'markowitz' in build_objective.__name__:
        tdy_covariance_path = covariances_path / f"covariances_{tdy.strftime('%y%m%d')}.parquet"
        symbols_tdy = tdy_data_updated.index
        tdy_covariance = pd.read_parquet(tdy_covariance_path).loc[symbols_tdy, symbols_tdy]
        tdy_covariance, symbols_wo_covariance_tdy = drop_na_matrix(tdy_covariance)
        if symbols_wo_covariance_tdy and logger is not None:
            policy = (
                f'On day {tdy}, the stocks {symbols_wo_covariance_tdy} have covariance data.'
                ' The weights on these stocks will be set to 0.0'
                ' (without looking at the signal or constraints).'
            )
            log(logger, policy, 'debug')
        tdy_data_updated = tdy_data_updated[~tdy_data_updated.index.isin(symbols_wo_covariance_tdy)]
        weights = pd.Series(data=0.0, index=symbols_wo_covariance_tdy, name='weight')
    return tdy_data_updated, weights, tdy_covariance

def get_weights_incompatible_boxes(tdy, tdy_data, logger):
    tdy_data = (
        tdy_data
        .eval('max_weight_tdy = weight_ytd + max_trade')
        .eval('min_weight_tdy = weight_ytd - max_trade')
    )

    incompatible = tdy_data.query('(max_weight < min_weight_tdy) | (min_weight > max_weight_tdy)')
    mask_right = incompatible['max_weight'] < incompatible['min_weight_tdy']
    mask_left = incompatible['min_weight'] > incompatible['max_weight_tdy']
    # assert (mask_right | mask_left).all()
    incompatible.loc[mask_right, 'weight'] = incompatible.loc[mask_right, 'min_weight_tdy']
    incompatible.loc[mask_left, 'weight'] = incompatible.loc[mask_left, 'max_weight_tdy']

    if not incompatible.empty and logger is not None:
        symbols = incompatible.index.tolist()
        policy = (
            f'On day {tdy}, the stocks {symbols} cannot satisfy the maximum weights'
            ' and the adv constraints simultaneously. The maximum trades to satisfy the'
            ' adv constraints and getting close to the maximum weights are performed'
            ' (without looking at the signal), violating the second constraint.'
        )
        log(logger, policy, 'info')
        cols = ['max_weight', 'min_weight', 'max_weight_tdy', 'min_weight_tdy']
        log(logger, incompatible.loc[:, cols], 'info')
    tdy_data_updated = tdy_data[~tdy_data.index.isin(incompatible.index)]
    tdy_data_updated.loc[:, 'max_weight_tdy'] = (
        tdy_data_updated[['max_weight_tdy', 'max_weight']].min(axis=1)
    )
    tdy_data_updated.loc[:, 'min_weight_tdy'] = (
        tdy_data_updated[['min_weight_tdy', 'min_weight']].max(axis=1)
    )
    assert (tdy_data_updated['min_weight_tdy'] <= tdy_data_updated['max_weight_tdy']).all()
    return tdy_data_updated.drop(columns=['max_weight', 'min_weight']), incompatible['weight']

def get_weights_small_positions(tdy, tdy_data_updated, weights_others, ptfo_params_updated, logger):
    index = weights_others.abs().le(ptfo_params_updated['min_abs_weight']).loc[lambda x: x].index
    tdy_data_updated = tdy_data_updated[~tdy_data_updated.index.isin(index)]
    weights = pd.Series(data=0.0, index=index, name='weight')
    if not index.empty and logger is not None:
        policy = (
            f'On day {tdy}, the stocks {index.tolist()} have weights under the `min_abs_weight`'
            f" value of {ptfo_params_updated['min_abs_weight']}. They will be set to 0."
        )
        log(logger, policy, 'info')
    return tdy_data_updated, weights

def update_ptfo_params(weights_incompatible_boxes, ptfo_params):
    ptfo_params_updated = ptfo_params.copy()
    if 'max_leverage' in ptfo_params:
        current_leverage = weights_incompatible_boxes.abs().sum()
        ptfo_params_updated['max_leverage'] -= current_leverage
        assert ptfo_params_updated['max_leverage'] > 0
    if 'leverage' in ptfo_params:
        current_leverage = weights_incompatible_boxes.abs().sum()
        ptfo_params_updated['leverage'] -= current_leverage
        assert ptfo_params_updated['leverage'] > 0
    if 'net_bounds' in ptfo_params:
        current_net = weights_incompatible_boxes.sum()
        ptfo_params_updated['net_bounds'] = [
            ptfo_params['net_bounds'][0] - current_net,
            ptfo_params['net_bounds'][1] - current_net
        ]
    return ptfo_params_updated

def build_boxes_constraints(tdy_data_updated, *variables):
    return [
        tdy_data_updated['min_weight_tdy'].to_numpy() <= variables[0],
        variables[0] <= tdy_data_updated['max_weight_tdy'].to_numpy()
    ]

def build_cross_sectional_constraints(ptfo_params_updated, *variables):
    constraints = []
    if 'net_bounds' in ptfo_params_updated:
        constraints += [
            ptfo_params_updated['net_bounds'][0] <= cp.sum(variables[0]),
            cp.sum(variables[0]) <= ptfo_params_updated['net_bounds'][1]
        ]
    if 'max_leverage' in ptfo_params_updated:
        constraints += [cp.sum(cp.abs(variables[0])) <= ptfo_params_updated['max_leverage']]
    if 'leverage' in ptfo_params_updated:
        constraints += [
            cp.sum(variables[1]) == 0.5 * ptfo_params_updated['leverage'],
            cp.sum(variables[2]) == 0.5 * ptfo_params_updated['leverage'],
        ]
    return constraints

def build_constraints(tdy_data_updated, ptfo_params_updated, *variables):
    # 0 is feasible
    constraints = [variables[0] == variables[1] - variables[2]]
    constraints += build_boxes_constraints(tdy_data_updated, *variables)
    constraints += build_cross_sectional_constraints(ptfo_params_updated, *variables)
    return constraints

def build_objective_alpha_tracker(
    tdy_data_updated, tdy_covariance, ptfo_params_updated, *variables
):
    del tdy_covariance, ptfo_params_updated
    weights = tdy_data_updated['weight'].to_numpy()
    objective = cp.norm(weights - variables[0], 2)
    return cp.Minimize(objective)

def build_costs(tdy_data_updated, ptfo_params_updated, *variables):
    weights = tdy_data_updated['weight'].to_numpy()
    expressions = {
        'transaction': (variables[0] - weights, 1),
        'shorting': (cp.neg(variables[0]), 1),
        'market_impact': (variables[0] - weights, 1.5),
    }
    costs = [
        (
            ptfo_params_updated['penalizations'][f'cost_{name}']
            * cp.sum(tdy_data_updated[f'cost_{name}'].to_numpy() @ cp.abs(expr) ** power)
        )
        for name, (expr, power) in expressions.items()
    ]
    return sum(costs)

def build_objective_alpha_tracker_with_costs(tdy_data_updated, _, ptfo_params_updated, *variables):
    weights = tdy_data_updated['weight'].to_numpy()
    objective = cp.norm(weights - variables[0], 2)
    costs = build_costs(tdy_data_updated, ptfo_params_updated, *variables)
    return cp.Minimize(objective + costs)

def build_objective_markowitz(
    tdy_data_updated, tdy_covariance, ptfo_params_updated, *variables,
):
    weights = tdy_data_updated['weight'].to_numpy()
    ret = weights @ variables[0]
    gamma = ptfo_params_updated['penalizations']['risk_aversion']
    risk = cp.quad_form(variables[0], tdy_covariance)
    costs = build_costs(tdy_data_updated, ptfo_params_updated, *variables)
    return cp.Minimize(-ret + gamma * risk + costs)

def build_and_solve_problem(
    tdy, tdy_data_updated, tdy_covariance, ptfo_params_updated, build_objective, *, logger,
):
    variables = [
        cp.Variable(len(tdy_data_updated)),
        cp.Variable(len(tdy_data_updated), nonneg=True),  # positive_part
        cp.Variable(len(tdy_data_updated), nonneg=True),  # negative_part
    ]
    constraints = build_constraints(tdy_data_updated, ptfo_params_updated, *variables)
    objective = build_objective(
        tdy_data_updated, tdy_covariance, ptfo_params_updated, *variables,
    )
    value, errored = solve_optimization_problem(objective, constraints, *variables, logger)
    if errored:
        policy = (
            f'On day {tdy}, the optimization problem failed to solve.'
            f' The weights will be filled forward.'
        )
        log(logger, policy, 'warning')
        weights = tdy_data_updated['weight_ytd']
    else:
        weights = pd.Series(data=value, index=tdy_data_updated.index)
    return weights.rename('weight'), errored

def pd_concat(objs, **kwargs):
    non_empty = [obj for obj in objs if not obj.empty]
    return pd.concat(non_empty, **kwargs) if non_empty else None

def get_weights(
    *, tdy, tdy_data, covariances_path, weights_ytd, ptfo_params,
    build_objective, logger=None,
):
    tdy_data = get_union_ytd_tdy(tdy_data, weights_ytd)

    # no optimization
    tdy_data_updated, weights_0 = get_weights_missing_market_data(tdy, tdy_data, logger)
    tdy_data_updated, weights_1, tdy_covariance = fetch_covariance(
        tdy, tdy_data_updated, covariances_path, build_objective, logger
    )
    tdy_data_updated, weights_2 = get_weights_incompatible_boxes(tdy, tdy_data_updated, logger)
    ptfo_params_updated = update_ptfo_params(weights_2, ptfo_params)

    # one optimization
    weights_others, errored =  build_and_solve_problem(
        tdy, tdy_data_updated, tdy_covariance, ptfo_params_updated, build_objective,
        logger=logger,
    )
    weights = pd_concat([weights_0, weights_1, weights_2, weights_others])

    # several optimizations
    if 'min_abs_weight' in ptfo_params_updated:
        weights_two = []
        errors = []
        while any(weights_others.abs().le(ptfo_params_updated['min_abs_weight'])):
            tdy_data_updated, weights_2 = get_weights_small_positions(
                tdy, tdy_data_updated, weights_others, ptfo_params_updated, logger
            )
            weights_two.append(weights_2)
            weights_others, error = build_and_solve_problem(
                tdy, tdy_data_updated, tdy_covariance, ptfo_params_updated, build_objective,
                logger=logger,
            )
            errors.append(error)
        errored = any([errored] + errors)
        weights = pd_concat([weights_0, weights_1, weights_2, weights_others] + weights_two)
    return weights, errored

def backtest(
    *, weights_and_market_data, covariances_path, ptfo_params,
    build_objective, logger=None,
):
    dates = weights_and_market_data.index.get_level_values(keys[0]).unique()
    assert dates.is_monotonic_increasing
    tdy = weights_and_market_data.dropna(subset='weight').index[0][0]

    dates = [date for date in dates if date >= tdy]
    weights = [None for _ in dates]
    errors = [None for _ in dates]

    # first day
    tdy_data = weights_and_market_data.loc[tdy]
    weights_ytd = pd.Series(0.0, index=tdy_data['weight'].dropna().index, name='weight_ytd')
    weights[0], errors[0] = get_weights(
        tdy=tdy,
        tdy_data=tdy_data,
        covariances_path=covariances_path,
        weights_ytd=weights_ytd,
        ptfo_params=ptfo_params,
        build_objective=build_objective,
        logger=logger,
    )
    # other days
    for ind, tdy in enumerate(dates[1:], start=1):
        log(logger, f'Running backtest on day: {tdy}.', 'trace')
        weights[ind], errors[ind] = get_weights(
            tdy=tdy,
            tdy_data=weights_and_market_data.loc[tdy],
            covariances_path=covariances_path,
            weights_ytd=weights[ind - 1],
            ptfo_params=ptfo_params,
            build_objective=build_objective,
            logger=logger,
        )
    return dates, weights, errors

def unique_lst_str(lst_str, sep='_'):
    unique_strings = list(OrderedDict.fromkeys(lst_str))
    return sep.join(unique_strings)

def flatten_multiindex(pd_df):
    cols = pd_df.columns.to_list()
    if isinstance(cols[0], tuple):  # multiindex
        pd_df.columns = [unique_lst_str(col) for col in cols]
    return pd_df

def pairs_cov(cols):
    return [
        pl.cov(*pair).alias(str(ind))
        for ind, pair in enumerate(combinations(cols, r=2))
    ]

def pairs_corr(cols, **kwargs):
    return [
        pl.corr(*pair, **kwargs).alias(str(ind))
        for ind, pair in enumerate(combinations(cols, r=2))
    ]

def condensed_to_square(condensed):
    dim = (1 + np.sqrt(1 + 8 * len(condensed))) / 2
    assert dim.is_integer()
    dim = int(dim)
    square = np.ones((dim, dim))
    upper_indices = np.triu_indices(dim, k=1)
    square[upper_indices] = square.T[upper_indices] = condensed
    return square

def np_square_to_pd_df(square, columns):
    frame = pd.DataFrame(square)
    frame.columns = frame.index = columns
    return frame

def shape_matrix(condensed, columns):
    matrix = condensed_to_square(condensed)
    return np_square_to_pd_df(matrix, columns)

def n_components_to_int(prop_expl_var, diag):
    return np.where(diag.cumsum() >= prop_expl_var * len(diag))[0][0] + 1

def clean_corr_mat(corr, n_components):
    res = None, None, None, None
    assert (
        (isinstance(n_components, int) and 1 <= n_components <= len(corr))
        or 0 < n_components < 1
    )
    try:
        uuu, diag, vvv = np.linalg.svd(corr, full_matrices=True)
        # uuu = vvv.T
        # corr = uuu @ np.diag(diag) @ vvv
    except np.linalg.LinAlgError:
        pass
    else:
        if 0 < n_components < 1:
            n_components = n_components_to_int(n_components, diag)
        diag[n_components:] = 0
        prop_expl_var = diag.mean()
        cleaned = proj_unit_diag(reduce(np.matmul, [uuu, np.diag(diag), vvv]))
        # components_.T * S**2 * components_ + sigma2 * eye(n_features)
        res = cleaned, n_components, uuu, prop_expl_var
    return res

def proj_unit_diag(mat):
    mat_ = mat.copy()
    np.fill_diagonal(mat_, 1)
    return mat_

def drop_na_matrix(corr_pd):
    corr_wo_nan = corr_pd.dropna(axis=1, how='all').dropna(axis=0, how='all')
    have_no_missing_value = sorted(corr_wo_nan.notna().all().loc[lambda x: x].index)
    return (
        corr_wo_nan.loc[have_no_missing_value, have_no_missing_value],
        sorted(set(corr_pd) - set(have_no_missing_value))
    )

def get_pca_factors(standardized_returns, n_components):
    corr = compute_corr_pd_to_pd(standardized_returns, True, method='spearman')
    corr_wo_nan, other_stocks = drop_na_matrix(corr)
    cleaned_corr = clean_corr_mat(corr_wo_nan.values, n_components)[0]
    # cleaned_corr, cut, uuu, _ = clean_corr_mat(corr_wo_nans.values, n_components)
    # if cut is not None and uuu is not None and uuu[:cut, :].size > 1:
    #     factors = np.matmul(standardized_returns, uuu[:cut, :].T)
    #     factors.columns = 'factor_' + factors.columns.astype('str')
    # else:
    #     factors = None
    axes = {'index': corr_wo_nan.index, 'columns': corr_wo_nan.index}
    return pd.DataFrame(cleaned_corr, **axes), other_stocks

def standardize(returns):
    means = returns.mean().rename('mean')
    vols = returns.std()
    return means, vols, (returns - means) / vols

# def get_residuals(standardized_returns, n_components):
#     factors, cleaned_corr, other_stocks = get_pca_factors(standardized_returns, n_components)
#     res = None, None, None, cleaned_corr, other_stocks
#     if factors is not None:
#         reg = LinearRegression(fit_intercept=False)
#         reg.fit(standardized_returns, factors)
#         # rets = means + vols * s_rets
#         # s_rets = loadings @ factors + residual_returns
#         # rets = means + vols * factors @ loadings + vols * residual_returns
#         cols = standardized_returns.columns
#         loadings = pd.DataFrame(data=reg.coef_, index=factors.columns, columns=cols)
#         residuals = standardized_returns.values - np.matmul(factors.values, loadings.values)
#         residual_returns = pd.DataFrame(
#             data=residuals, index=standardized_returns.index, columns=cols
#         )
#         res = factors, loadings, residual_returns, cleaned_corr, other_stocks
#     return res

def destandardize(loadings, residual_returns, vols):
    return loadings * vols, residual_returns * vols

def get_pca_factors_model(returns, n_components):
    means, vols, standardized_returns = standardize(returns)
    cleaned_corr, other_stocks = get_pca_factors(  # get_residuals
        standardized_returns, n_components
    )
    # loadings, residual_returns = destandardize(loadings_, residual_returns_, vols)
    return {
        'means': means,
        'vols': vols,
        # 'factors': factors,
        # 'loadings': loadings,
        # 'residual_returns': residual_returns,
        'cleaned_corr': cleaned_corr,
        'other_stocks': other_stocks,
    }

def compute_corr_pd_to_pd(returns_pd, through_polars=True, **kwargs):
    if not through_polars:
        corr = returns_pd.corr(**kwargs)
    else:
        columns = list(returns_pd)
        returns_pl = pl.DataFrame(returns_pd)
        condensed = returns_pl.select(pairs_corr(columns, **kwargs))
        corr = shape_matrix(condensed.row(0), columns)
    return corr

def get_data_for_backtest(
    weights, ptfo_params, *,
    path_market_data='prices_yf.parquet',
    volume_smoother=smoother_volumes,
    cost_transaction=transaction_cost('volume_smoothed'),
    cost_shorting=COST_SHORTING,
    cost_market_impact=market_impact_cost('volume_smoothed', 'volatility')
):
    start_date = weights.get_column(keys[0]).min()
    end_date = weights.get_column(keys[0]).max()
    market_data = load_and_prepare_market_data(
        path_market_data=path_market_data,
        start_date=start_date,
        volume_smoother=volume_smoother,
        cost_transaction=cost_transaction,
        cost_shorting=cost_shorting,
        cost_market_impact=cost_market_impact,
    ).filter(pl.col(keys[0]) <= end_date).collect()

    weights_and_market_data = (
        market_data
        .join(weights, how='left', on=keys)
        .select(*keys, 'price', 'volume_smoothed', 'weight', pl.col('^cost.*$'))
        # 'price' is not used for the backtest
    )
    full_index = get_prod_uniques(weights_and_market_data, keys).sort(keys)
    weights_and_market_data = full_index.join(weights_and_market_data, on=keys, how='left')
    weights_and_market_data = weights_and_market_data.with_columns(
        max_trade=pl.col('volume_smoothed') * ptfo_params['prop_adv'] / ptfo_params['aum'],
        min_weight=pl.lit(ptfo_params['weight_bounds'][0]),
        max_weight=pl.lit(ptfo_params['weight_bounds'][1]),
    )
    return weights_and_market_data

def optimized_weights_and_errors_to_pd(dates, weights, errors):
    optimized_weights = (
        pd.concat([weight.to_frame().assign(date=date) for date, weight in zip(dates, weights)])
        .reset_index()
        .sort_values(keys)
        .set_index(keys)
    )
    optimized_weights_ytd = (
        pd.concat([weight.to_frame().assign(date=date) for date, weight in zip(dates[1:], weights)])
        .reset_index()
        .sort_values(keys)
        .set_index(keys)
        .rename(columns={'weight': 'optimized_weight_ytd'})
    )

    errors_mapped = [{True: 0, False: np.nan}[error] for error in errors]
    errors = pd.Series(errors_mapped, index=dates, name='errored')
    return optimized_weights, optimized_weights_ytd, errors

def prepare_data_for_backtest_check(
    weights_and_market_data, optimized_weights, optimized_weights_ytd
):
    quality_check = optimized_weights.copy().rename(columns={'weight': 'optimized_weight'})
    quality_check = (
        weights_and_market_data
        .join(quality_check, how='outer', on=keys)
        .join(optimized_weights_ytd, how='outer', on=keys)
        .sort_values(keys)
    )
    quality_check.loc[:, 'optimized_weight_abs'] = quality_check['optimized_weight'].abs()
    quality_check.eval('weight_diff = optimized_weight - optimized_weight_ytd', inplace=True)
    quality_check.loc[:, 'turnover'] = quality_check['weight_diff'].abs()
    return quality_check

def plot_errors_and_references(pd_series, name, errors, *refs):
    # TODO: use plot_pd_frame
    to_plot = pd_series.to_frame(name).join(errors, how='left', on=keys[0])
    plt.figure()
    to_plot[name].plot()
    to_plot['errored'].plot(marker='*', color='r', grid=True)
    for ref in refs:
        plt.axhline(ref, color='r')
    plt.show()

def plot_analytics(weights_and_market_data, plots=(
    'coverage', 'description', 'leverage', 'turnover', 'autocorrelation', 'ic', 'pnl'
)):
    # TODO: add ic decay, ts ic for different horizons, etc.
    weights_and_market_data_w_forward_return = (
        weights_and_market_data
        .with_columns(forward_return=pl.col('price').pct_change().shift(-1).over(keys[1]))
    )

    weights_ = weights_and_market_data_w_forward_return.select(*keys, 'weight')
    market_data_ = weights_and_market_data_w_forward_return.select(
        *keys, 'price', 'forward_return', pl.col('^cost.*$')
    )

    attrs = ['sum', 'mean', 'median', 'std']
    counts_n_sums = (
        weights_
        .with_columns(turnover=turnover)
        .group_by(keys[0], maintain_order=True)
        .agg(
            [
                pl.len(),
                pl.col('weight').is_not_null().sum().alias('n_weights'),
                (pl.col('weight').is_not_null().sum() / pl.len()).alias('prop'),
                pl.col('weight').abs().sum().alias('leverage'),
                pl.col('turnover').sum(),
            ]
            + [
                getattr(pl.col('weight'), attr)().alias(f'{attr}_weight')
                for attr in attrs
            ]
        )
    )
    if 'coverage' in plots:
        for col in ['len', 'n_weights', 'prop']:
            plot_ts(counts_n_sums.select(keys[0], col), keys[0])

    if 'description' in plots:
        pivoted = pivot_to_pd(weights_, *keys, 'weight')
        plot_desc_pfo_matrix(pivoted)

    if 'leverage' in plots:
        plot_ts(counts_n_sums.select(keys[0], 'leverage'), keys[0], with_mean=True)

    if 'turnover' in plots:
        plot_ts(counts_n_sums.select(keys[0], 'turnover'), keys[0], with_mean=True)

    if 'autocorrelation' in plots:
        shifts = list(range(100))
        autocorrelation = (
            weights_
            .with_columns([
                pl.col('weight').shift(shift).over(keys[1]).alias(str(shift))
                for shift in shifts
            ])
            .select([
                pl.corr(str(shift), 'weight', method='spearman')
                for shift in shifts
            ])
            .to_pandas()
            .T
        )
        autocorrelation.index = autocorrelation.index.astype(int)
        autocorrelation.columns = ['autocorrelation']
        plot_pd_frame(autocorrelation)

    if 'ic' in plots:
        sub = (
            weights_and_market_data_w_forward_return
            .group_by(keys[0], maintain_order=True)
            .agg(ic=pl.corr('weight', 'forward_return', method='spearman'))
        )
        plot_ts(sub.select(keys[0], 'ic'), keys[0], with_mean=True)

        info_coeff = (
            weights_and_market_data_w_forward_return
            .select(pl.corr('weight', 'forward_return', method='spearman'))
            .item()
        )
        print('ic', info_coeff)

    if 'pnl' in plots:
        pnl, metrics = compute_metrics(weights_, market_data_)
        plot_ts(pnl.select(keys[0], 'cumsum', 'cummax'), keys[0])
        print(metrics.row(0, named=True))

def plot_backtest_quality_checks(quality_check, errors, ptfo_params):
    # correlation weight and optimized weight
    wo_nans = quality_check.loc[:, ['weight', 'optimized_weight']].dropna(axis=0, how='all')
    transfer_coeff = wo_nans.corr(method='spearman').iloc[0, 1]
    print(transfer_coeff)

    correlations = wo_nans.groupby('date').corr().iloc[0::2, -1]
    correlations.index = correlations.index.droplevel(level=1)
    plot_pd_frame(correlations)

    # leakage of positions
    ## every weight has an optimized weight
    mask = quality_check['weight'].notna() & quality_check['optimized_weight'].isna()
    assert quality_check[mask].empty

    ## coming from leaking
    mask = quality_check['optimized_weight'].notna() & quality_check['weight'].isna()
    leaking = quality_check[mask].groupby('date').agg({
        'optimized_weight': ['count', 'sum'],
        'optimized_weight_abs': 'sum',
    })
    leaking = flatten_multiindex(leaking)
    for col in leaking.columns:
        plot_errors_and_references(leaking[col], col, errors)

    ## not coming from leaking
    mask = quality_check['optimized_weight'].notna() & quality_check['weight'].notna()
    not_leaking = quality_check[mask].groupby('date').agg({
        'optimized_weight': ['count', 'sum'],
        'optimized_weight_abs': 'sum',
    })

    not_leaking = flatten_multiindex(leaking)
    for col in not_leaking.columns:
        plot_errors_and_references(leaking[col], col, errors)

    # counts
    n_positions = (
        quality_check
        ['optimized_weight']
        .dropna()
        .groupby('date')
        .count()
    )
    plot_errors_and_references(n_positions, 'n_positions', errors)

    n_positions = quality_check['optimized_weight'].dropna()
    counts = []
    for threshold in np.logspace(-5, -1, 5):
        counts.append((
            n_positions[n_positions.abs().ge(threshold)]
            .groupby('date')
            .count()
            .rename(round(np.log(threshold) / np.log(10), 2))
        ))
    pd.concat(counts, axis=1).plot(grid=True)

    leverage_one = quality_check['optimized_weight'].dropna()
    leverage_one = leverage_one.groupby('date').transform(lambda x: x / x.abs().sum())
    # leverage_one.groupby('date').apply(lambda x: x.abs().sum()).plot(grid=True)

    n_effective = (
        leverage_one
        .pow(2)
        .groupby('date')
        .sum()
        .rdiv(1)
        .clip(None, 300)
    )
    plot_errors_and_references(n_effective, 'n_effective', errors)

    # check that constraints are satisfied
    checks = [
        ['max_leverage', quality_check['optimized_weight_abs'].groupby('date').sum()],
        ['leverage', quality_check['optimized_weight_abs'].groupby('date').sum()],
        ['min_abs_weight', (
            quality_check['optimized_weight_abs']
            .loc[lambda x: x > 0]
            .clip(0, 2 * ptfo_params.get('min_abs_weight', float('inf')))
            .groupby('date')
            .min())
        ],
        ['weight_bounds', quality_check['optimized_weight'].groupby('date').min()],
        ['weight_bounds', quality_check['optimized_weight'].groupby('date').max()],
        ['prop_adv', quality_check['adv_traded'].groupby('date').max()],
        ['net_bounds', quality_check['optimized_weight'].groupby('date').sum()],
    ]

    for constraint, data in checks:
        if constraint in ptfo_params:
            print(constraint)
            constraints = ptfo_params[constraint]
            if not isinstance(constraints, list):
                constraints = [constraints]
            plot_errors_and_references(data, constraint, errors, *constraints)
