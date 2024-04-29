from nearcorrmat import __version__
from itertools import combinations
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from nearcorrmat.functions import *

def test_version():
    assert __version__ == '0.1.0'


def test_is_def_pos(mat=None, **kwargs):
    mat = gen_sym_psd(100, 1) if mat is None else mat
    assert_allclose(mat.T, mat, **kwargs)  # assert_equal
    assert np.linalg.cholesky(mat).any()


def test_unit_diag(mat=None):
    mat = gen_sym_psd(100, 2) if mat is None else mat
    assert_equal(np.diag(proj_unit_diag(mat)), np.ones(len(mat)))


def test_conj():
    mat = gen_sym_psd(100, 3)
    corr = conj(np.diag(np.diag(mat) ** -0.5), mat)
    test_unit_diag(corr)
    test_is_def_pos(corr)


def test_real_eig():
    mat = gen_sym_psd(100, 4)
    diag, t_mat = real_eig(mat)
    assert_equal(diag.real, diag)
    assert_equal(t_mat.real, t_mat)
    assert_allclose(conj(t_mat, diag), mat)


def test_proj_cone():
    projected = proj_cone(gen_sym_psd(100, 5))
    test_is_def_pos(projected)


def test_dual_to_primal():
    rng = np.random.default_rng(6)
    mat = rng.standard_normal((100, 100))
    assert_equal(np.diag(dual_to_primal(-np.diag(mat), mat)), np.zeros(100))


def test_grad_theta():
    dim = 100
    mat = gen_sym_psd(dim, 7)
    diag = np.diag(mat)
    corr = mat * np.outer(diag, diag) ** -.5
    assert_allclose(grad_theta(corr), [0] * dim, atol=1e-12)


def test_grad_theta_true():
    dim = 100
    mat = gen_sym_psd(dim, 12)
    diag = np.diag(mat)
    corr = mat * np.outer(diag, diag) ** -.5
    zero = grad_theta_true(np.zeros(dim), corr)
    assert_allclose(zero, [0] * dim, atol=1e-12)


def test_projs_via_nearest_corr():
    atol = 1e-12
    proj_laplace = nearest_corr(laplace, 'bfgs', 1e-2 * atol, 10000)[0]
    def test_proj(method):
        projected, errors = nearest_corr(laplace, method, atol, 40)
        assert_allclose(projected, proj_laplace, atol=atol)
        return np.array(errors)

    errs = [test_proj(method) for method in
            ['grad', 'admm_v0', 'admm_v1', 'bfgs', 'l_bfgs', 'newton']]
    errs = [err if err.ndim == 1 else err[:, 0] for err in errs]
    assert all((
        len(err_1) != len(err_2) or max(err_1 - err_2) >= 1e-3 * atol)
               for err_1, err_2 in combinations(errs, 2))


def dual_grad(mat, err_max, ite_max):
    '''Gradient descent for the dual objective,
    this is checked to be the exact same algorithm as `proj_grad`,
    by checking that the errors match.

    '''
    dim = len(mat)
    var_dual = np.ones(dim) - np.diag(mat)
    grad_theta_old = grad_theta_true(var_dual, mat)
    err = err_max + 1
    ite = 0
    errs = list()
    while err > err_max and ite < ite_max:
        ## step 1: dual descent
        var_dual = var_dual - grad_theta_old

        ## step 2: update and error computation
        grad_theta_new = grad_theta_true(var_dual, mat)
        err = norm(grad_theta_old)
        grad_theta_old = grad_theta_new.copy()
        errs.append(err)
        ite += 1
    return proj_cone(dual_to_primal(var_dual, mat)), errs


def test_dual_grad():
    err_max = 1e-6
    ite_max = 1000
    mat = gen_sym_psd(20, 11)
    errs_proj_grad = proj_grad(mat, err_max, ite_max)[1]
    errs_dual_grad = proj_grad(mat, err_max, ite_max)[1]
    assert_equal(errs_proj_grad, errs_dual_grad)


def test_update_admm():
    mat = gen_sym_psd(100, 0)
    prim_2 = mat.copy()
    dual = np.zeros(mat.shape)
    prim_1_1, prim_2_new_1 = update_admm(mat, 1, 1, prim_2, dual)
    prim_1_0, prim_2_new_0 = update_admm(mat, 0, 1, prim_2, dual)
    test_unit_diag(prim_1_1)
    test_unit_diag(prim_2_new_0)
    test_is_def_pos(prim_2_new_1 + 1e-10 * np.eye(100))
    test_is_def_pos(prim_1_0 + 1e-10 * np.eye(100))
    assert norm(prim_2_new_1 + prim_2_new_0) <= 2 * norm(mat)
    # as both projection have a triple norm of 1


def test_update_rho_admm():
    dual = np.ones(10)
    case_0 = update_rho_admm(1, dual, 1, 10, 2, 5)
    assert_equal(case_0[0], 0.5)
    assert_equal(case_0[1], 2 * dual)
    case_1 = update_rho_admm(1, dual, 10, 1, 3, 5)
    assert_equal(case_1[0], 3)
    assert_equal(case_1[1], dual / 3)
    case_2 = update_rho_admm(1, dual, 10, 1, 3, 10)
    assert_equal(case_2[0], 1)
    assert_equal(case_2[1], dual)


def test_update_bfgs():
    dim = 100
    inv_hess = np.eye(dim)
    rng = np.random.default_rng(8)
    dvar = rng.standard_normal(dim)
    dgrad = np.abs(rng.standard_normal(dim)) * np.sign(dvar)
    update = update_bfgs(inv_hess, dvar, dgrad)
    bfgs_mat = inv_hess + update
    assert np.linalg.matrix_rank(update) == 2
    assert_allclose(bfgs_mat @ dgrad, dvar)
    test_is_def_pos(bfgs_mat)


def compare_bfgs_updates(mat, memory=5):
    dim = len(mat)
    var_dual = np.zeros(dim)
    var_dual_l = np.zeros(dim)
    grad_theta_old = grad_theta(mat)

    inv_hess = np.eye(dim)
    dvars = []
    dgrads = []
    grads_bfgs = [grad_theta_old]
    grads_l_bfgs = [grad_theta_old]

    for ite in range(10):
        direc = inv_hess @ grads_bfgs[-1]
        if ite >= memory:
            direc_l = inv_hess_limited(grads_l_bfgs[-1], dvars, dgrads)
        else:
            direc_l = direc

        var_dual = var_dual - direc
        var_dual_l = var_dual_l - direc_l

        grad_theta_new = grad_theta_true(var_dual, mat)
        grad_theta_new_l = grad_theta_true(var_dual_l, mat)

        diff_grads = grad_theta_new - grads_bfgs[-1]
        diff_grads_l = grad_theta_new_l - grads_l_bfgs[-1]

        inv_hess = inv_hess + update_bfgs(inv_hess, -direc, diff_grads)

        if ite >= memory:
            del dvars[0]
            del dgrads[0]
        dvars.append(-direc_l)
        dgrads.append(diff_grads_l)
        grads_bfgs.append(grad_theta_new)
        grads_l_bfgs.append(grad_theta_new_l)
    return grads_bfgs, grads_l_bfgs


def test_inv_hess_limited():
    mat = gen_sym_psd(100, 9)
    grads_bfgs, grads_l_bfgs = compare_bfgs_updates(mat)
    assert_equal(grads_bfgs[:6], grads_l_bfgs[:6])
    assert_allclose(grads_bfgs[6:], grads_l_bfgs[6:], atol=1e-3)


def test_proj_newton():
    mat = build_newton_mat(gen_sym_psd(100, 10))
    test_is_def_pos(mat, atol=1e-15)
    # the convergence of proj_newton,
    # tests the quality of the direction found by inverting this matrix
