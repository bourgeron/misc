from pathlib import Path
from ftplib import FTP
from loguru import logger
import polars as pl

directory = Path('./nasdaq/')
directory.mkdir(exist_ok=True)

# Download listed symbols: ftp.nasdaqtrader.com/Symboldirectory/nasdaqlisted.txt
FILE_NAME = 'nasdaqlisted.txt'
local_file = directory / FILE_NAME

ftp = FTP('ftp.nasdaqtrader.com')
ftp.login()
logger.info(ftp.nlst())
ftp.cwd('Symboldirectory')
logger.info(ftp.nlst())
with open(local_file, 'wb') as file:
    ftp.retrbinary(f'RETR {FILE_NAME}', file.write)
ftp.quit()

# Convert local file to parquet
lines = Path(local_file).read_text(encoding='utf-8').split('\n')
lines = [line.split('|') for line in lines]
listed = pl.DataFrame(lines[1:-3], schema=lines[0])
parquet_name = f"{lines[-2][0].split(': ')[-1]}_{local_file.stem}.parquet"
listed.write_parquet(local_file.parent / parquet_name)
logger.info(lines[-1])
