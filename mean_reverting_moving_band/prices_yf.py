import polars as pl
from utils import keys, cast_to_date, are_keys, fold, get_full_index_per_symbol

min_n_points = 252
max_null_prop = 0.02

null_prop = pl.col('price').null_count() / pl.len()

prices = (
    pl.scan_parquet('yf/yf.parquet')
    .rename({'Date' : keys[0], 'Symbol': keys[1], 'Adj Close': 'price', 'Volume': 'volume'})
    .select(*keys, 'price', 'volume')
    .collect()
)
prices = cast_to_date(prices, keys[0])
assert prices.null_count().sum_horizontal().item() == 0
assert are_keys(prices, keys)[0]

cleaner = prices

full_index_per_symbol = get_full_index_per_symbol(fold(cleaner, *keys), *keys)
cleaner = full_index_per_symbol.join(cleaner, how='left', on=keys)

cleaner = (
    cleaner
    .filter(pl.col('price').is_null().all().not_().over(keys[0]))
    .filter((pl.col('price') >= 0).all().over(keys[1]))
    .filter(pl.col('price').is_null().not_().sum().over(keys[1]) >= min_n_points)
)

cleaner_wo_weekends = (
    cleaner
    .filter(pl.col('date').dt.weekday().is_in([6, 7]).not_())
    .filter(null_prop.over(keys[1]) <= max_null_prop)
)

symbols = cleaner_wo_weekends.select(pl.col('symbol').unique())
cleaner = symbols.join(cleaner, on='symbol', how='inner').select(*keys, pl.exclude(keys))
cleaner = cleaner.with_columns(pl.col('volume') * pl.col('price'))
cleaner.write_parquet('prices_yf.parquet')
