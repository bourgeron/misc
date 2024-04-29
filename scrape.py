from itertools import tee, islice
import polars as pl
from generic import consume, loop_finite

def expansion_bisection(get_and_extract, infinite_iterable):
    expansion, bisection = tee(infinite_iterable, 2)

    # expansion
    prev_data = pl.DataFrame()
    path_n_data = get_and_extract(next(expansion))
    data = path_n_data[1]
    ind = 0
    retrieved = [[ind, path_n_data]]
    expand = 1
    while not prev_data.frame_equal(data):
        prev_data = data
        _, data = get_and_extract(next(expansion))
        ind += 1
        retrieved.append([ind, path_n_data])
        consume(expansion, expand)
        ind += expand
        expand *= 2
    del retrieved[-1]

    # bisection
    if len(retrieved) > 1:
        inds, paths_n_data = zip(*retrieved)
        upb = inds[-1]
        lwb = inds[-2]
        bisection = list(islice(bisection, upb + 1))
        prev_data = paths_n_data[-1][1]
        while upb - lwb > 1:
            mid = lwb + (upb - lwb) // 2
            path_n_data = get_and_extract(bisection[mid])
            data = path_n_data[1]
            if prev_data.frame_equal(data):
                upb = mid
            else:
                lwb = mid
                retrieved.append([mid, path_n_data])
        inds, paths_n_data = zip(*retrieved)
        paths_n_data = list(paths_n_data)
        lst = [ele for ind, ele in enumerate(bisection) if ind not in inds]
    else:
        paths_n_data = [retrieved[0][1]] if retrieved else []
        lst = []

    return paths_n_data, lst

def crawl_infinite(get_and_extract, infinite_iterable, processes, *, with_ray):
    paths_n_data, lst = expansion_bisection(get_and_extract, infinite_iterable)
    paths_n_data.extend(loop_finite(get_and_extract, lst, processes, with_ray=with_ray))
    return paths_n_data
