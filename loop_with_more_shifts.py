from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import polars as pl
from ccp import construct_stat_arb
from utils import (
    set_logger, loop_finite, trade_multi,
    get_universe_with_start_date, get_splits, join_many,
    load_and_prepare_market_data, are_keys
)

set_logger(logger, ['INFO', 'WARNING'], './logs')

keys = ['date', 'symbol']

PROCESSES = 9
WITH_RAY = False
N_REPEAT = 10

def change_seed(seed):
    return construct_stat_arb(seed=seed, **args)

def fit_multi(formation, seeds):
    args['prices'] = formation
    return loop_finite(change_seed, seeds, PROCESSES, with_ray=WITH_RAY)

def fit_n_trade(formation, trading, seeds, t_max):
    stat_arbs = fit_multi(formation, seeds)
    return trade_multi(stat_arbs, formation, trading, t_max)

P_MAX = 100
SPREAD_MAX = 1
MOVING_MEAN = True
SOLVER = 'CLARABEL'

args = {
    'prices': None,
    'P_max': P_MAX,
    'spread_max': SPREAD_MAX,
    'moving_mean': MOVING_MEAN,
    # 's_init': None,  # set to default
    # 'mu_init': None,  # set to default
    # 'seed': None,  # changed
    # 'M': None,  # not used at all!
    'solver': SOLVER,  # default value
    # 'second_pass': False,  # set to default
    'logger': logger,
}

path_simus = Path('./simus_with_more_shifts')
path_simus.mkdir(parents=True, exist_ok=True)

def one_simulation(params):
    universe = get_universe_with_start_date(params)
    path = Path(params['path'])
    universe.write_parquet(path.parent / f'universe_{path.stem}.parquet')

    splits = get_splits(universe)
    seeds_all = [
        [np.random.randint(9999) for _ in range(N_REPEAT)]
        for _ in range(len(splits))
    ]
    n_nulls_weights = [
        fit_n_trade(formation, trading, seeds, params['t_max'])
        for seeds, (formation, trading) in zip(seeds_all, splits)
    ]
    _, weights = zip(*n_nulls_weights)
    assert len(weights) == len(splits)
    cleaned_weights = [wghts for wghts in weights if wghts is not None]

    pd_frame = (
        pd.concat(cleaned_weights)
        .reset_index()
        .melt(id_vars=('date', 'attempt'), var_name=keys[1], value_name='weight')
        .loc[:, [*keys, 'attempt', 'weight']]
    )

    schema = {
        'date': pl.Date,
        'attempt': pl.UInt8,
        'symbol': pl.Utf8,
        'weight': pl.Float64
    }

    min_position = 1e-5
    one_run = (
        pl.DataFrame(pd_frame)
        .cast(schema)
        .drop_nulls('weight')
        .filter(pl.col('weight').abs().ge(min_position))
        .with_columns(path=pl.lit(str(path)))
    )
    assert are_keys(one_run, [*keys, 'attempt'])[0]
    one_run.write_parquet(path)

divs = np.array([50, 25, 17, 12, 6, 4, 2])

loop = {
    'formation_length': (252 / divs).astype(int),
    't_max': (252 / divs).astype(int),
    'n_stocks': [10, 100, 1000],
}
index = [pl.DataFrame(vals, schema=[name]) for name, vals in loop.items()]
index = (
    join_many(index, how='cross')
    .filter(pl.col('formation_length') >= pl.col('t_max'))
    .filter(pl.col('formation_length') > 21)
)

UPDATE_FREQUENCY = 21

start_dates = (
    load_and_prepare_market_data()
    .select(start_date='date')
    .unique(maintain_order=True)
    .head(400)
    .with_columns(index=pl.lit(1))
    .with_columns(is_update_date=(pl.col('index').cum_sum() - 1) % UPDATE_FREQUENCY == 0)
    .drop('index')
    .collect()
)

EXIT_LENGTH = 21

total_lengths = (
    index
    .with_columns(total_length=pl.col('t_max') + EXIT_LENGTH + pl.col('formation_length'))
)
assert total_lengths.get_column('total_length').max() <= len(start_dates)
heads = [
    part.drop('total_length').join(start_dates.head(total_length), how='cross')
    for part in total_lengths.partition_by('total_length')
    if (total_length := part.get_column('total_length').unique().item())
]

path_name = pl.concat_str([
    pl.lit(path_simus.stem), pl.lit('/'), pl.col('path'), pl.lit('.parquet')
])
index = (
    pl.concat(heads)
    .filter(pl.col('is_update_date'))
    .drop('is_update_date')
    .with_columns(path=pl.lit(1))
    .with_columns((pl.col('path').cum_sum() - 1).cast(pl.Utf8))
    .with_columns(path=path_name)
)
# mu_memory not an argument of construct_stat_arb
# mu_memory defaults to 21 in _State.init, cf. lines 79 and 186 of ccp.py

index.write_parquet(path_simus / 'index.parquet')

for pars in index.iter_rows(named=True):
    logger.warning(pars)
    try:
        one_simulation(pars)
    except ValueError as exception:
        logger.warning(f"{exception} at {pars}")
