from pathlib import Path
from loguru import logger
import numpy as np
import polars as pl
from utils import (
    set_logger, keys, get_index_loop,
    read_weights_one_hyperparameter, smooth_and_normalize,
    loop_finite, end_date_train
)

set_logger(logger, ['INFO', 'WARNING'], './logs')

# write index

path_preprocessing = Path('./preprocessing')
path_preprocessing.mkdir(parents=True, exist_ok=True)

path_simus = Path('./simus_with_more_shifts')
path_index_simus = path_simus / 'index.parquet'
index_simus = (
    pl.read_parquet(path_index_simus)
    .select('formation_length', 't_max', 'n_stocks')
    .unique()
)

divs = np.array([126, 84, 63, 50, 25, 17, 12, 6, 4, 2])
loop = {
    'first_step': ['do_nothing', 'get_leverage_one'],
    'window': (252 / divs).astype(int),
    'third_step': ['do_nothing', 'rank_scale'],
    # 'fourth_step': 'mean_zero' + 'leverage one'
}
index_preprocessing = get_index_loop(loop)
index_preprocessing = index_simus.join(index_preprocessing, how='cross')
assert index_preprocessing.null_count().sum_horizontal().item() == 0
logger.warning(len(index_preprocessing))

path_name = pl.concat_str([
    pl.lit(path_preprocessing.stem), pl.lit('/'), pl.col('path_preprocessing'), pl.lit('.parquet')
])
index_preprocessing = (
    index_preprocessing
    .with_columns(pl.lit(1).alias('path_preprocessing'))
    .with_columns((pl.col('path_preprocessing').cum_sum() - 1).cast(pl.Utf8))
    .with_columns(path_name.alias('path_preprocessing'))
)
index_preprocessing.write_parquet(path_preprocessing / 'index.parquet')

# function for each iteration

def preprocess(params):
    logger.warning(params)
    hyperparameter = {
        'formation_length': params.pop('formation_length'),
        't_max': params.pop('t_max'),
        'n_stocks': params.pop('n_stocks'),
    }
    weights = (
        read_weights_one_hyperparameter(hyperparameter, path_index_simus)
        .filter(pl.col(keys[0]) <= end_date_train)
        .collect()
    )
    weights = smooth_and_normalize(weights, params)
    (
        weights
        .with_columns(path_preprocessing=pl.lit(str(params['path_preprocessing'])))
        .write_parquet(params['path_preprocessing'])
    )

# go through the loop
loop_finite(preprocess, index_preprocessing.iter_rows(named=True), 1, with_ray=False)

# loop_finite(preprocess, index_preprocessing.iter_rows(named=True), None, with_ray=True)
