from itertools import combinations, accumulate
from functools import reduce
import numpy as np
from numpy.linalg import norm
import polars as pl
import sklearn
import sklearn.preprocessing

def preprocessor_fit(spread, preprocessor_name, **kwargs):
    imports = ['preprocessing', preprocessor_name]
    preprocessor = reduce(getattr, imports, sklearn)(**kwargs)
    values = spread.to_numpy().reshape(-1, 1)
    preprocessor.fit(values)
    return preprocessor

def preprocessor_transform(preprocessor, spread):
    values = spread.to_numpy().reshape(-1, 1)
    preprocessed = preprocessor.transform(values).reshape(-1)
    return pl.Series(preprocessed).rename(spread.name)

def form_spread(frame, pair, coef, sep='___'):
    return (
        frame
        .select([pl.col(p) * c for p, c in zip(pair, coef)])
        .sum(axis=1)
        .rename(sep.join(pair))
    )

def form_pl_frame(lst_dics, cols):
    data = [
        {col: val for col, val in dic.items() if col in cols}
        for dic in lst_dics
    ]
    return pl.DataFrame(data, schema=cols)

def trading_rule(cur_pos, st_spread, threshold):
    new_pos = 0
    if st_spread < -threshold:
        new_pos = +1
    elif st_spread > threshold:
        new_pos = -1
    elif np.sign(st_spread) * cur_pos == -1:
        new_pos = cur_pos
    return new_pos

def fit(formation, pair, preprocessor_name, sep='___', **kwargs):
    # assert not any(sep in col for col in formation.columns)
    assert formation.select(pl.col(f'^.*{sep}.*$')).is_empty()
    pair_name = sep.join(pair)
    first_line = formation.select(pair).row(0)
    coef = [1 / first_line[0], -1 / first_line[1]]
    spread = (
        formation
        .select([pl.col(p) * c for p, c in zip(pair, coef)])
        .sum(axis=1)
        .rename(pair_name)
    )
    preprocessor = preprocessor_fit(spread, preprocessor_name, **kwargs)
    score = norm(spread)
    return {
        'pair_name': pair_name,
        'pair': pair,
        'coef': coef,
        'spread': spread,
        'preprocessor': preprocessor,
        'score': score,
    }

def positions_one_spread(trading, record, rule):
    pair_name = record['pair_name']

    spread = form_spread(trading, record['pair'], record['coef'])
    st_spread = preprocessor_transform(record['preprocessor'], spread)
    positions = accumulate(st_spread.to_numpy(), rule, initial=0)
    positions = (
        pl.DataFrame(list(positions)[1:], schema=[pair_name])
        .select([
            (c * pl.col(pair_name)).alias(p)
            for c, p in zip(record['coef'], record['pair'])])
    )
    return (
        pl.concat([trading.select('date'), positions], how='horizontal')
        .melt(id_vars='date', variable_name='asset', value_name='position')
        .sort(['date', 'asset'])
    )


def to_unit_leverage(frame, col):
    return (
        frame
        .with_columns(pl.col(col) / pl.col(col).abs().sum().over('date'))
    )


def window_describe(col, attrs=None):
    if attrs is None:
        attrs = ('count', 'null_count', 'mean', 'std', 'min', 'max', 'median')
    return [getattr(pl.col(col), attr)().alias(attr) for attr in attrs]


def simple_agg(lst_frames, keys):
    concatenated = pl.concat(lst_frames)
    return (
        concatenated
        .groupby(keys)
        .agg(pl.exclude(keys).sum())
        .sort(keys)
    )   


def trade(trading, records, rule):
    positions = [
        positions_one_spread(trading, record, rule)
        for record in records
    ]
    return simple_agg(positions, ['date', 'asset'])


def fit_trade(formation, preprocessor_name, top, trading, rule, **kwargs):
    pairs = combinations(formation.drop('date').columns, r=2)
    fitted = [fit(formation, pair, preprocessor_name, **kwargs) for pair in pairs]

    scores = form_pl_frame(fitted, ['pair_name', 'score']).sort('score')
    pairs_names = scores.head(top).get_column('pair_name').to_list()
    records = [desc for desc in fitted if desc['pair_name'] in pairs_names]

    return trade(trading, records, rule)

def plot_ts(t_s):
    t_s.sort('date').to_pandas().set_index('date').plot(grid=True)
