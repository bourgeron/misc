from datetime import timedelta
from pathlib import Path
from loguru import logger
import pandas as pd
from utils import (
    set_logger, load_and_prepare_market_data,
    keys, init_date, get_pca_factors_model,
    loop_finite
)

N_COMPONENTS = 0.70

set_logger(logger, ['INFO', 'WARNING'], './logs')

path_covariance_matrices = Path('./covariance_matrices')
path_covariance_matrices.mkdir(parents=True, exist_ok=True)

period = timedelta(days=730)  # 2 years

# Load dates

dates = (
    load_and_prepare_market_data(start_date=init_date - period)
    .select('date')
    .unique()
    .sort('date')
    .collect()
    .get_column('date')
)
logger.warning(f'n_dates: {len(dates)}')

# Loop

def ad_hoc_pl(date):
    date_str = date.strftime('%y%m%d')
    path_pca = path_covariance_matrices / ('pca_' + date_str + '.parquet')
    if not path_pca.exists():
        logger.warning(date)  # PicklingError is ray is used, to comment
        returns_pd = pd.read_parquet(
            path_covariance_matrices / ('returns_' + date_str + '.parquet')
        ).set_index(keys[0])
        res = get_pca_factors_model(returns_pd, N_COMPONENTS)
        cleaned_corr = res['cleaned_corr']
        if cleaned_corr is not None:
            pd.DataFrame(res['other_stocks']).to_parquet(
                path_covariance_matrices / ('other_stocks_' + date_str + '.parquet')
            )
            res['vols'].to_frame().to_parquet(
                path_covariance_matrices / ('vols_' + date_str + '.parquet')
            )
            cleaned_corr.to_parquet(path_pca)
        else:
            logger.warning(date)  # PicklingError is ray is used, to comment

# go through the loop
# loop_finite(ad_hoc_pl, dates, None, with_ray=True)
loop_finite(ad_hoc_pl, dates, 1, with_ray=False)
