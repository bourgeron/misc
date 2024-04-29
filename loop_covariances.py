from datetime import timedelta
from pathlib import Path
from loguru import logger
import polars as pl
from utils import (
    set_logger,
    keys, init_date, load_and_prepare_market_data,
    pairs_cov, shape_matrix,
    loop_finite
)

set_logger(logger, ['INFO', 'WARNING'], './logs')

path_covariance_matrices = Path('./covariance_matrices')
path_covariance_matrices.mkdir(parents=True, exist_ok=True)

period = timedelta(days=730)  # 2 years

# this snippet is going to overload the RAM:
# rolling_covs = (
#     returns
#     .rolling(keys[0], period=period)
#     .agg(pairs_cov(columns))
#     .sort(keys[0])
# )
# instead, do it by hand, on small universes
# first: load all the symbols
# second: for each day, compute the covariance matrix

# Load the symbols

path_simus = Path('./simus_with_more_shifts/')

paths_universes = (
    pl.scan_parquet(path_simus / 'index.parquet')
    .filter(pl.col('n_stocks') <= 100)
    .select('path')
    .collect()
)

symbols = [
    set(
        pl.scan_parquet(Path(path).parent / ('universe_' + Path(path).name))
        .select('symbol')
        .unique()
        .collect()
        .get_column('symbol')
        .to_list()
    )
    for path in paths_universes.get_column('path')
]
columns = sorted(set.union(*symbols))
logger.warning(f'n_symbols: {len(columns)}')

# Load returns

returns = (
    load_and_prepare_market_data(start_date=init_date - period)
    .with_columns(pl.col('price').pct_change().over(keys[1]).alias('return'))
    .filter(pl.col(keys[1]).is_in(columns))
    .collect()
    .pivot(index=keys[0], columns=keys[1], values='return', sort_columns=True)
    .sort('date')
)
returns.select('date').write_parquet(path_covariance_matrices / 'dates.parquet')
dates = returns.get_column('date')
logger.warning(f'n_dates: {len(dates)}')

# Loop

def ad_hoc_pl(date):
    path_covariances = (
        path_covariance_matrices / ('covariances_' + date.strftime('%y%m%d') + '.parquet')
    )
    if not path_covariances.exists():
        logger.warning(date)  # PicklingError is ray is used, to comment
        sub = (
            returns
            .filter(pl.col('date') <= date)
            .filter(pl.col('date') > date - period)
        )
        sub.write_parquet(
            path_covariance_matrices / ('returns_' + date.strftime('%y%m%d') + '.parquet')
        )

        shape_matrix(
            sub
            .select(pairs_cov(columns))
            .row(0),
            columns
        ).to_parquet(path_covariances)

# 5x faster but crashes the machine... silently
# def ad_hoc_pd(date):
#     logger.warning(date)  # PicklingError is ray is used, to comment
#     name = date.strftime('%y%m%d') + '.parquet'
#     path = path_covariance_matrices / name
#     sub = (
#         returns
#         .filter(pl.col('date') <= date)
#         .filter(pl.col('date') > date - period)
#         .to_pandas()
#         .set_index('date')
#     )
#     sub.cov().to_parquet(path)

# go through the loop
# loop_finite(ad_hoc_pl, dates, None, with_ray=True)
loop_finite(ad_hoc_pl, dates, 1, with_ray=False)
