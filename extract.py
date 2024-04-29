from io import StringIO
import json
import pandas as pd
import polars as pl
from generic import log

def extract_all_contents(extractor, directory, pattern, /, logger=None, **kwargs):
    for path in directory.glob(pattern):
        extractor(path, logger=logger, **kwargs)

def extractor_quandl(path, /, key, logger=None):
    # An `extractor` takes a:
    # pathlib.Path, and a logger (generally, as named argument and defaulting to None).
    # It can take other named arguments or kwargs.
    # It saves a file, at a path `out_path`.
    # It returns the couple (out_path, frame) where frame is a Polars or Pandas DataFrame,
    # having the constant column 'path', with (string) value `out_path`.
    data_key = pl.DataFrame()
    data = json.loads(path.read_text())
    log(logger, data['meta'])
    parquet_path = path.with_suffix('.parquet')
    if not (data_key := pl.DataFrame(data[key])).is_empty():
        data_key = data_key.with_columns(path=pl.lit(str(parquet_path)).cast(pl.Utf8))
        data_key.write_parquet(parquet_path)
    return parquet_path, data_key

def extract_tables(text, /, logger=None, **kwargs):
    try:
        pd_dfs = pd.read_html(StringIO(text), **kwargs)
    except ValueError as error:
        log(logger, f'Error {error} extracting tables', 'warning')
        pd_dfs = []
    return pd_dfs

def _extract_fields(box, fields):
    name = box.columns[0]
    fields_names, substrings, patterns, dtypes = zip(*fields)

    subs = '(' + '|'.join(substrings) + ')'
    extracted = (
        box
        .filter(pl.col(name).str.contains(subs))
        .with_columns(
            pl.col(name).str.extract(subs)
            .replace(dict(zip(substrings, fields_names)))
        )
        .group_by(name)
        .first()
        .transpose(column_names=name)
    )

    base = (
        pl.DataFrame([], schema=['name', *fields_names])
        .with_columns(pl.all().cast(pl.Utf8))
    )
    return (
        pl.concat([base, extracted], how='diagonal')  # fills by nulls
        .with_columns(pl.col('name').fill_null(name))
        .with_columns([
            pl.col(field_name).str.extract(pattern)
            for field_name, pattern in zip(fields_names, patterns)
        ])
        .with_columns([
            pl.col(field_name).cast(dtype)
            for field_name, dtype in zip(fields_names, dtypes)
        ])
    )

def extractor_first_infobox_fields(path, /, logger=None, **kwargs):
    assert 'fields' in kwargs, "The 'fields' parameter must be provided."
    data = None
    text = path.read_text()
    boxes = extract_tables(text, logger=logger, attrs={'class': 'infobox'})
    if boxes is not None and (boxes := [b for b in boxes if b.shape[1] == 2]):
        box = pl.DataFrame(boxes[0]).with_columns(pl.all().cast(pl.Utf8))
        data = (
            _extract_fields(box, kwargs['fields'])
            .with_columns(path=pl.lit(str(path)).cast(pl.Utf8))
        )
        data.write_parquet(path.with_suffix('.parquet'))
    else:
        log(logger, f'There is no infobox at {path}')
    return data
