from itertools import combinations
import polars as pl

def get_date_fields(key_date, fields=None):
    if fields is None:
        fields = ('hour', 'minute', 'second', 'millisecond', 'microsecond', 'nanosecond')
    return [getattr(pl.col(key_date).dt, field)().alias(field) for field in fields]

def not_finite_floats_to_none(col):
    return (
        pl.when(pl.col(col).is_nan() | pl.col(col).is_infinite())
        .then(None).otherwise(pl.col(col))
        .alias(col)
    )

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
