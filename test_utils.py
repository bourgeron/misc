from numpy.testing import assert_allclose
from utils import *


def test_gen_sym_psd(mat=None, **kwargs):
    mat = gen_sym_psd(100, 1) if mat is None else mat
    assert_allclose(mat.T, mat, **kwargs)  # assert_equal
    assert np.linalg.cholesky(mat).any()

def test_lod_2_dol():
    lod = [
        {'blue': 1, 'err': 8, 'green': 2},
        {'blue': 3, 'err': 7, 'green': 5},
        {'blue': 4, 'err': 6, 'green': 6},
        {'blue': -2, 'err': 4, 'green': -3},
    ]
    dol = lod_2_dol(lod)
    assert isinstance(dol, dict) and len(dol) == 3 and len(dol['blue']) == 4
    return dol

def test_fit_log_errs():
    dol = test_lod_2_dol()
    lin_fits = fit_log_errs(pd.DataFrame(dol))
    assert all((
        isinstance(lin_fits, pd.DataFrame),
        list(lin_fits.index) == ['slope', 'intercept'],
        sorted(lin_fits) == sorted(dol),
        lin_fits['blue'].isnull().sum() == lin_fits['green'].isnull().sum() == 2,))

def test_gen_expes_dims(seed=0):
    expes = gen_expes_dims(0, 5, 10, 20, seed)
    assert all((
        isinstance(expes, dict),
        min(expes.keys()) == 1,
        max(expes.keys()) == int(1e5),
        len(expes) == 10,
        all(len(lst) == 20 for lst in expes.values()),
        all(len(vec) == 1 for vec in expes[1]),
    ))

def test_time_one_func(seed=1):
    func = lambda x: x
    expes = gen_expes_dims(0, 5, 10, 20, seed)
    time = time_one_func(func, expes)
    assert all((
        isinstance(time, pd.DataFrame),
        list(time.index) == ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'],
        sorted(time) == sorted(expes),
    ))
