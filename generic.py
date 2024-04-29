from collections import deque, OrderedDict
from functools import reduce
from itertools import combinations, islice
from multiprocessing import Pool
from pathlib import Path
from sys import stderr
import ray

def check_partition(full_set, parts):
    # parts is a list of sets
    assert all(bool(part) for part in parts)  # each part is not empty
    non_empty = [
        intersection for left, right in combinations(parts, r=2)
        if (intersection := left & right)
    ]
    assert not non_empty, non_empty  # parts are pairwise disjoint
    union = set.union(*parts)  # the union of the parts is full_set
    assert union == full_set, (full_set - union, union - full_set)

def del_dict(dic, dic_keys):
    for key in dic_keys:
        del dic[key]

def replace_chars(word, table):
    return reduce(lambda string, char: string.replace(*char), table, word)

def first_argm(lst_tpls, how):
    return how(lst_tpls, key=lambda x: x[1])[0] if lst_tpls else None

def log(logger, msg, level='info'):
    if logger is not None:
        getattr(logger, level)(msg)

def level_only(level):
    def inner(record):
        return record['level'].name == level
    return inner

def set_logger(logger, ordered_levels, logs_path):
    format_ = '{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}'
    logs_path = Path(logs_path)
    logs_path.mkdir(parents=True, exist_ok=True)
    logger.remove()
    for level in ordered_levels:
        logger.add(
            logs_path / f'log_{level.lower()}.log',
            filter=level_only(level),
            delay=True,
            format=format_
        )
    logger.add(logs_path / 'log_full.log', level=ordered_levels[0], delay=False, format=format_)
    logger.add(stderr, level=ordered_levels[-1], format=format_)

def consume(iterator, inte=None):
    '''Advance the iterator inte-steps ahead. If inte is None, consume entirely.
    https://docs.python.org/3/library/itertools.html#itertools-recipes'''
    # Use functions that consume iterators at C speed.
    if inte is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position inte
        next(islice(iterator, inte, inte), None)

def loop_finite(fun, finite_iterable, processes, /, with_ray=True):
    # TODO: pass `processes` to `ray`
    if processes == 1:
        out = [fun(ele) for ele in finite_iterable]
    elif not with_ray:
        with Pool(processes) as pool:
            out = list(pool.imap_unordered(fun, finite_iterable))
    else:
        ray.init()
        out = ray.get([ray.remote(fun).remote(ele) for ele in finite_iterable])
        ray.shutdown()
    return out

def unique_lst_str(lst_str, sep='_'):
    unique_strings = list(OrderedDict.fromkeys(lst_str))
    return sep.join(unique_strings)

def have_common_type(lst, typ):
    return all(isinstance(ele, typ) for ele in lst)

def check_common_type(lst, typs):
    types = {typ: have_common_type(lst, typ) for typ in typs}
    assert any(types.values())
    return [typ for typ, bol in types.items() if bol][0]
