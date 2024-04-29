from pathlib import Path
from ftplib import FTP
from loguru import logger
import polars as pl
import yfinance as yf

save_path = Path('./yf/')
save_path.mkdir(exist_ok=True)


# Download listed symbols from nasdaq

## Fetch: ftp.nasdaqtrader.com/Symboldirectory/nasdaqlisted.txt
FILE_NAME = 'nasdaqlisted.txt'
local_file = save_path / FILE_NAME

ftp = FTP('ftp.nasdaqtrader.com')
ftp.login()
logger.info(ftp.nlst())
ftp.cwd('Symboldirectory')
logger.info(ftp.nlst())
with open(local_file, 'wb') as file:
    ftp.retrbinary(f'RETR {FILE_NAME}', file.write)
ftp.quit()

## Convert to parquet
lines = Path(local_file).read_text(encoding='utf-8').split('\n')
lines = [line.split('|') for line in lines]
listed = pl.DataFrame(lines[1:-3], schema=lines[0])
parquet_name = f"{lines[-2][0].split(': ')[-1]}_{local_file.stem}.parquet"
listed.write_parquet(local_file.parent / parquet_name)
logger.info(lines[-1])


# Download yahoo finance historical data

## Load symbols
# path_last_nasdaq = max(save_path.glob('*.parquet'))
path_last_nasdaq = local_file.parent / parquet_name
listed = pl.scan_parquet(path_last_nasdaq).select('Symbol').collect()

## Loop through symbols
for symbol in listed.to_series():
    logger.info(symbol)
    data = yf.download(symbol).assign(Symbol=symbol)
    data.to_parquet(save_path / (symbol + '.parquet'))

## Dump all data as a parquet file
keys = ('Date', 'Symbol')
all_data = (
    pl.scan_parquet(save_path.glob('[A-Z]*.parquet'))
    .sort(keys)
    .select(pl.col(keys), pl.exclude(keys))
)
logger.info(
    all_data.select(
        pl.col('Date').n_unique(),
        pl.col('Symbol').n_unique()
    ).collect()
)
all_data.collect().write_parquet(save_path / 'yf.parquet')
