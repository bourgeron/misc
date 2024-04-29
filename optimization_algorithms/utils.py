from timeit import repeat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gen_sym_psd(int_n, seed=None, int_p=None):
    '''Generates a random symmetric positive semidefinite array.

    Parameters
    ----------
    int_n : int
        Size of the output array.
    seed : {None, int}, optional
        Random seed used by `numpy.random.default_rng`.
    int_p : {None, int}, optional
        Almost surely the rank of the output matrix is `min(int_p, int_n)`.

    Returns
    -------
    mat : array
        Random symmetric positive semidefinite array.

    '''
    if int_p is None:
        int_p = int_n
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((int_n, int_p))
    return np.matmul(draws, draws.T)

def lod_2_dol(lod):
    '''Convert list of dicts to dict of lists.

    Parameters
    ----------
    lod : list[dict]
        List of dictionaries.

    Returns
    -------
    dol : dict
        Dictionary of lists.

    '''
    return {key: [dic[key] for dic in lod] for key in lod[0]}

# def errs_2_log_errs(errs):
#     '''Convert dict of errs to pandas DataFrame of log errs.'''
#     frame_errs = pd.DataFrame(errs)
#     log_errs = dict()
#     for col in frame_errs:
#         vals = frame_errs[col].iloc[:-1]
#         if 'err' in col:
#             log_errs[col] = np.log(vals - frame_errs[col].iloc[-1])
#         else:
#             log_errs[col] = np.log(vals)
#     return pd.DataFrame(log_errs)

def fit_log_errs(errs, plot=False):
    '''Fit lines to log errors.

    Parameters
    ----------
    errs : pandas.DataFrame
        Errors (whose log is taken).
    plot : bool, optional
        Whether to plot the erros or not.

    Returns
    -------
    lin_fits : pandas.DataFrame
        Slopes and intercepts.

    '''
    log_errs = errs.apply(np.log)
    ites = np.array(log_errs.index)
    lin_fits = log_errs.apply(lambda vals: np.polyfit(ites, vals, 1))
    lin_fits.index = ['slope', 'intercept']
    if plot:
        plt.figure()
        for col in log_errs:
            lin_reg = lin_fits.at['slope', col] * ites + lin_fits.at['intercept', col]
            plt.plot(log_errs[col], label=f'{col}')
            plt.plot(ites, lin_reg, label=f'{col} lin reg')
        plt.grid()
        plt.legend()
        plt.show()
    return lin_fits

def gen_expes_dims(exp_start, exp_stop, n_dims, n_runs, seed=None):
    '''Generate of dictionary {dim: list[array]} for log-spaced dimensions `dim`.

    Parameters
    ----------
    exp_start, exp_stop : float
        Start and end exponents.
    n_dims : int
        Number of dimensions.
    n_runs : int
        Length of the list for each dimension.

    Returns
    -------
    expes : dict
        Dictionary {dim: list[array]} for log-spaced dimensions `dim`.

    '''
    dims = np.logspace(exp_start, exp_stop, n_dims).astype(int)
    rng = np.random.default_rng(seed)
    return {
        dim: [rng.standard_normal(dim) for _ in range(n_runs)]
        for dim in dims}

def time_one_func(func, expes, n_repeats=5, n_execs=10):
    '''Generate of dictionary {dim: list[array]} for log-spaced dimensions `dim`.

    Parameters
    ----------
    func : float
        Function taking a 1-d array as input, whose performance is tested.
    expes : int
        Dictionary {dim: list[array]} for log-spaced dimensions `dim`.
    n_repeats : int, optional
        Repeat count passed to the Timer.
    n_execs : int, optional
        Number of executions passed to the Timer.

    Returns
    -------
    expes : pandas.DataFrame
        Time measurements.

    '''
    times = {
        dim: [
            repeat(lambda: func(imput), number=n_execs, repeat=n_repeats)
            for imput in imputs]
        for dim, imputs in expes.items()}
    times_frame = pd.concat([
        pd.DataFrame(times_).stack().reset_index(drop=True).rename(dim)
        for dim, times_ in times.items()], axis=1)
    return times_frame.describe()
