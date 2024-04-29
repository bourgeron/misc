from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import polars as pl
from utils import (
    set_logger, keys, get_index_loop,
    read_data_one_preprocessing, get_data_for_backtest,
    backtest, build_objective_alpha_tracker_with_costs,
    compute_metrics,
    loop_finite,
)

set_logger(logger, ['INFO', 'WARNING'], './logs')

ptfo_params = {
    'aum': 100e6,
    'max_leverage': 5,
    # 'leverage': 5,  # does not do what is aimed at
    # 'min_abs_weight': 1e-4,
    # not convex, could increase the computing time a lot, but, does not in practice
    'weight_bounds': [-0.02, 0.02],
    'prop_adv': 0.02,
    'net_bounds': [-0.01, 0.01],
}

# write index

path_alpha_tracker = Path('./alpha_tracker')
path_alpha_tracker.mkdir(parents=True, exist_ok=True)

loop = {
    'cost_transaction': np.logspace(-5, 5, 10),
    'cost_shorting': np.logspace(-5, 5, 10),
    'cost_market_impact': [0.0],
}
index_alpha_tracker = get_index_loop(loop)
assert index_alpha_tracker.null_count().sum_horizontal().item() == 0
logger.warning(len(index_alpha_tracker))

path_name = pl.concat_str([
    pl.lit(path_alpha_tracker.stem), pl.lit('/'), pl.col('path_alpha_tracker'), pl.lit('.parquet')
])
index_alpha_tracker = (
    index_alpha_tracker
    .with_columns(pl.lit(1).alias('path_alpha_tracker'))
    .with_columns((pl.col('path_alpha_tracker').cum_sum() - 1).cast(pl.Utf8))
    .with_columns(path_name.alias('path_alpha_tracker'))
)
index_alpha_tracker.write_parquet(path_alpha_tracker / 'index.parquet')

# function for each iteration

path_preprocessing = Path('preprocessing/1576.parquet')

_, weights = read_data_one_preprocessing(path_preprocessing, './preprocessing/index.parquet')
weights = weights.collect()

weights_n_data = get_data_for_backtest(weights, ptfo_params)

weights_and_market_data = weights_n_data.to_pandas().set_index(keys)

def preprocess(params):
    logger.warning(params)

    ptfo_params['penalizations'] = params
    dates, optimized_weights, _ = backtest(
        weights_and_market_data=weights_and_market_data,
        covariances_path=None,
        ptfo_params=ptfo_params,
        build_objective=build_objective_alpha_tracker_with_costs,
        logger=logger,
    )
    lst = [weight.to_frame().assign(date=date) for date, weight in zip(dates, optimized_weights)]
    optimized_weights = (
        pl.DataFrame(pd.concat(lst).reset_index())
        .select(pl.col('date').cast(pl.Date()), 'symbol', 'weight')
        .sort(keys)
    )

    weights_and_market_data_w_forward_return = (
        weights_n_data
        .drop('weight')
        .join(optimized_weights, on=keys, how='left')
        .sort(keys)
        .with_columns(forward_return=pl.col('price').pct_change().shift(-1).over(keys[1]))
    )
    weights_ = weights_and_market_data_w_forward_return.select(*keys, 'weight')
    market_data_ = weights_and_market_data_w_forward_return.select(
        *keys, 'price', 'forward_return', pl.col('^cost.*$')
    )
    pnl, _ = compute_metrics(weights_, market_data_)

    (
        pnl
        .with_columns(path_alpha_tracker=pl.lit(str(params['path_alpha_tracker'])))
        .write_parquet(params['path_alpha_tracker'])
    )

# go through the loop
loop_finite(preprocess, index_alpha_tracker.iter_rows(named=True), 1, with_ray=False)

# loop_finite(preprocess, index_alpha_tracker.iter_rows(named=True), None, with_ray=True)
