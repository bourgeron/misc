from itertools import accumulate
import numpy as np
import pandas as pd

def gen_rand_cov_mat(size):
    rng = np.random.default_rng()
    mat = rng.standard_normal((size, size))
#     mat = mat / np.linalg.norm(mat, axis=0)
    return np.matmul(mat.T, mat)

def generate_increments(matu, n_steps, cov, seed):
    n_steps_ = n_steps - 1
    dtime = matu / n_steps_
    time = np.linspace(0, matu, num=n_steps)
    rng = np.random.default_rng(seed)
    dbrowns = rng.multivariate_normal(
        mean=[0] * len(cov),
        cov=dtime * cov,
        size=n_steps_,
        check_valid='raise')
    return dtime, time, dbrowns

def homogeneous(scheme, inits, dtime, time, dbrowns, /, **kwargs):
    'dX_t / X_t = a(t) dt + b(t) dW_t'
    returns = scheme(dtime, time[:-1].reshape(-1, 1), dbrowns, **kwargs)
    rets_n_init = np.insert(1 + returns, 0, inits, axis=0)
    return np.cumprod(rets_n_init, axis=0)

def path_independent(scheme, inits, dtime, time, dbrowns, /, **kwargs):
    'dX_t = a(t) dt + b(t) dW_t'
    diffs = scheme(dtime, time[:-1].reshape(-1, 1), dbrowns, **kwargs)
    diffs_n_init = np.insert(diffs, 0, inits, axis=0)
    return np.cumsum(diffs_n_init, axis=0)

def time_independent(scheme, inits, dtime, time, dbrowns, /, **kwargs):
    'dX_t = a(X_t) dt + b(X_t) dW_t; scheme is in fact time independent, here for compatibility'
    dbrowns_n_init = np.insert(dbrowns, 0, inits, axis=0)
    def recursion(x_t, dbrown):
        return x_t + scheme(dtime, time, dbrown, x_t, **kwargs)
    return np.array(list(accumulate(dbrowns_n_init, recursion)))

def generic(scheme, inits, dtime, time, dbrowns, /, **kwargs):
    'dX_t = a(t, X_t) dt + b(t, X_t) dW_t'
    dbrowns_n_init = np.insert(dbrowns, 0, inits, axis=0)
    def recursion(time_x_t, time_dbrown):
        return (
            time_x_t[0],
            time_x_t[1] + scheme(dtime, *time_dbrown, time_x_t[1], **kwargs)
        )
    paths = accumulate(zip(np.insert(time, 0, 0), dbrowns_n_init), recursion)
    return np.array(list(zip(*paths))[1])

def generate_path_euler(
    scheme, scheme_form, inits, matu, n_steps, /,
    cov=None, seed=None, **kwargs):
    if cov is None:
        cov = np.eye(len(inits))
    else:
        assert len(cov) == len(inits), "incompatible inputs' sizes"
    dtime, time, dbrowns = generate_increments(matu, n_steps, cov, seed)
    paths = scheme_form(scheme, inits, dtime, time, dbrowns, **kwargs)
    return time, paths

def _format_cols(cols):
    max_chars = len(str(cols.max()))
    return [str(col).zfill(max_chars) for col in cols]

def gen_membership(xs_len, prop):
    dates = pd.bdate_range('2010', '2020')
    data = np.random.rand(len(dates), xs_len) < prop
    membership = pd.DataFrame(index=dates, data=data)
    membership.columns = _format_cols(membership.columns)
    return membership

def gen_data(membership):
    data = np.random.randn(*membership.shape)
    axes = {axis: getattr(membership, axis) for axis in ['index', 'columns']}
    return pd.DataFrame(data=data, **axes).where(membership)
