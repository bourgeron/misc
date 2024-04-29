from datetime import timedelta
from pathlib import Path
from loguru import logger
import polars as pl
from utils import (
    set_logger, keys, init_date, get_prod_uniques,
    load_and_prepare_market_data,
    read_data_one_preprocessing, compute_metrics,
    loop_finite, end_date_train
)

set_logger(logger, ['INFO', 'WARNING'], './logs')

path_simus = Path('./preprocessing/')
path_index = path_simus / 'index.parquet'

# load market_data in memory

market_data = (
    load_and_prepare_market_data()
    .with_columns(pl.lit(0.0).alias('cost_transaction'), pl.lit(0.0).alias('cost_shorting'))  # no cost
    .with_columns(forward_return=pl.col('price').pct_change().shift(-1).over(keys[1]))
    .filter(pl.col('date') >= init_date)
    .filter(pl.col('date') <= end_date_train + timedelta(days=3))
)

# may be unnecessary
full_index = get_prod_uniques(market_data, keys).sort(keys)
market_data = full_index.join(market_data, on=keys, how='left').collect()

def drop_metrics(path):
    new_path = path.parent / ('metric_' + path.name)
    if not new_path.exists():
        params, weights = read_data_one_preprocessing(path, path_index)
        logger.warning(params)
        _, metrics = compute_metrics(weights.collect(), market_data)
        metrics.with_columns(path_preprocessing=pl.lit(str(path))).write_parquet(new_path)

# go through the loop
loop_finite(drop_metrics, path_simus.glob('[0-9]*.parquet'), 1, with_ray=False)

# loop_finite(drop_metrics, path_simus.glob('[0-9]*.parquet'), None, with_ray=True)
