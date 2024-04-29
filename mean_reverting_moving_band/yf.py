from pathlib import Path
from loguru import logger
import yfinance as yf
import polars as pl

save_path = Path('./yf/')
save_path.mkdir(exist_ok=True)

# Load last symbols from nasdaq
path_last_nasdaq = max(Path('./nasdaq/').glob('*.parquet'))
listed = pl.scan_parquet(path_last_nasdaq).select('Symbol').collect()

# Download yahoo finance historical data
for symbol in listed.to_series().to_list():
    logger.info(symbol)
    data = yf.download(symbol).assign(Symbol=symbol)
    data.to_parquet(save_path / (symbol + '.parquet'))

# Load all data
keys = ('Date', 'Symbol')
all_data = (
    pl.scan_parquet(save_path.glob('*.parquet'))
    .sort(keys)
    .select(pl.col(keys), pl.exclude(keys))
    .collect()
)
assert all_data.null_count().sum_horizontal().item() == 0
logger.info(all_data.select(pl.col('Date').n_unique(), pl.col('Symbol').n_unique()))
logger.info(all_data.shape)
all_data.write_parquet(save_path / 'yf.parquet')
