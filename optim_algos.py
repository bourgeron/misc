import numpy as np
from numpy.linalg import norm, inv  # for caching
import pandas as pd


# bisection algo

def bisection_increasing(func, point, lbd, ubd, tol=1e-9):
    '''Bisection algorithm for the (1d) increasing function `func`.
    Solves: find x : func(x) = point.

    Parameters
    ----------
    func : function
        One-d array, point at which the value is sought.
    point : float
        The zeros of func - point are sought.
    lbd, ubd : float
        Initial lower and upper bounds for the solution.
    tol : float, optional
        Error on the solution. Defaults to 1e-9.

    Returns
    -------
    mid : float
        Solution of the equation.
    ite : int
        Number of iterations performed.

    '''
    assert (diffb := ubd - lbd) > 0 and tol > 0
    ite_max = np.log2(diffb / tol)
    ite = 0
    mid = 0.5 * (ubd + lbd)
    while (ubd - lbd > tol and ite < ite_max):
        mid = lbd + 0.5 * (ubd - lbd)  # the halved sum may overflow
        y_mid = func(mid) - point
        if y_mid < 0:
            lbd = mid
        elif y_mid > 0:
            ubd = mid
        else:
            lbd = mid
            ubd = mid
        ite += 1
    return mid, ite

def init_prox_lp(point, expo, cst, lam, tol=1e-9):
    '''Initialize `prox_lp_1d*` algorithms.

    Parameters
    ----------
    point : numpy.array
        One-d array, point at which the value is sought.
    expo : float
        Exponent: expo >= 1.
    cst, lam : float
        Non-negative parameters. As the functional, the result depends only on their product.
        The constant `cst` is here for easy generalization to the multi-d setting.
    tol : float, optional
        Values whose absolute value is less than `tol` are treated as 0. Defaults to 1e-9.

    Returns
    -------
    point_0 : numpy.array
        Initial value to be passed to `prox_lp_1d*` algorithm.

    '''
    abs_point = abs(point)
    lamc = lam * cst
    if min(abs(lamc), abs_point / lamc) <= tol:
        point_0 = 0
    elif (abs_point - 1) * (expo - 2) > 0:
        point_0 = np.sign(point) * (abs_point / lamc) ** (1 / (expo - 1))
    else:
        point_0 = point
    return point_0

def prox_lp_1d_newton_raphson(point, expo, cst, lam, point_0=None):
    '''Proximal operator of the following function at point:
    x |-> lam * cst * 1 / expo * abs(x) ^ expo, in dimension 1.
    Computed by a Newton-Raphson method.
    Same as `prox_lp_1d_bisection`, for expo >= 1.5, but 1.5x faster.

    Parameters
    ----------
    point : numpy.array
        One-d array, point at which the value is sought.
    expo : float
        Exponent: expo > 1.5.
    cst, lam : float
        Non-negative parameters. As the functional, the result depends only on their product.
        The constant `cst` is here for easy generalization to the multi-d setting.
    point_0 : numpy.array or None, optional
        Initial parameter, 1-d array. Defaults to: init_prox_lp(point, expo, cst, lam).

    Returns
    -------
    sol : numpy.array
        Proximal operator at `point`.

    '''
    assert cst >= 0 and lam >= 0 and expo > 1.5
    if point_0 is None:
        val = init_prox_lp(point, expo, cst, lam)
    val_old = val
    ite = 0
    tol = 1e-9
    err = tol + 1
    if abs(point) < tol:
        val = 0
    else:
        while (err > tol and ite < 20):
            res = lam * cst * abs(val) ** (expo - 2)
            val = (
                (expo - 2) * res * val + point) / ((expo - 1) * res + 1)
            ite += 1
            err = abs(val - val_old)
            val_old = val
    return val, ite


# optimization algos for ptfos problems

def init_algo(mat, vec, maxes, maxes_default, **options):
    '''Initialize variables for any algo.

    Parameters
    ----------
    mat : numpy.array
        Input real positive semidefinite array, for the objective function.
    vec : numpy.array
        Input array to project.
    maxes : dict
        Dictionary whose keys are among ['er', 'eac', 'ite']. It can be {}.
        These keys stand for: `relative error`, `absolute constraint error`, `iteration`.
    maxes_default : dict
        Default values for maxes.
    **options : dict, optional
        Extra arguments.

    Returns
    -------
    init_vars : dict
        Initialized variables.

    '''
    errors = sorted(set(maxes_default) - {'ite'})
    assert 'ite' in maxes_default and len(errors) >= 1, (
        'Default parameters for `ite` and at least one type of error must be provided.')
    maxes_default.update(maxes)
    init_vars = {
        'mat': mat,
        'mat_sym': None if mat is None else 0.5 * (mat + mat.T),
        'vec': vec,
        'dim': len(vec),
        'errs': [],
        'ite': 0,
        'errors': errors,
        'seed': 0,  # random seed
    }
    assert mat.shape == (init_vars['dim'], init_vars['dim']) and vec.shape == (init_vars['dim'],), (
        'The array `mat` must be square, `vec` a 1d array and they must have compatible shapes: '
        + f'{mat.shape} vs {vec.shape} do not satisfy these conditions.')

    init_vars.update({f'max_{key}': val for key, val in maxes_default.items()})
    for error in init_vars['errors']:
        init_vars[error] = 1 + init_vars[f'max_{error}']
    init_vars.update(options)
    return init_vars

def keep_iterating(vrbls):
    '''Determine if the criteria to stop iterating are satisfied or not.

    Parameters
    ----------
    vrbls : dict
        Current variables.

    Returns
    -------
    keep_iterating : bool
        Whether to continue to iterate or not.

    '''
    return (
        all((vrbls[error] > vrbls[f'max_{error}'] for error in vrbls['errors']))
        and vrbls['ite'] < vrbls['max_ite'])

def nnls_ccd(mat, vec, maxes, **options):
    '''Cyclic gradient coordinate descent for non negative least squares:
    min._x   ||mat @ x - vec||_2   s. t.   x >= 0.
    https://www.stat.cmu.edu/~ryantibs/convexopt-F18/lectures/coord-desc.pdf

    Parameters
    ----------
    mat : numpy.array
        Input real positive semidefinite array, for the objective function.
    vec : numpy.array
        Input array.
    maxes : dict
        Dictionary whose keys are ['er', 'ite'].
        These keys stand for: `relative error`, `iteration`.
    **options : dict, optional
        The random seed `seed` can be passed.

    Returns
    -------
    sol : numpy.array
        Solution of the problem.
    errors : pandas.DataFrame
        Successive relative (for each coordinate) errors for the sup norm.
    vrbls : dict
        Variables used in the optimization.

    '''
    vrbls = init_algo(mat, vec, maxes, {'er': 1e-10, 'ite': 10000}, **options)
    vrbls.update({
        'norms': norm(mat, axis=0),
        'ers': [vrbls['er'] + 1 for _ in range(vrbls['dim'])],
        'beta': np.random.default_rng(vrbls['seed']).standard_normal(vrbls['dim']),
        'res': vrbls['vec'] - np.matmul(vrbls['mat'], vrbls['beta']),
    })
    vrbls['sq_norms'] = vrbls['norms'] ** 2
    while keep_iterating(vrbls):
        coor = vrbls['ite'] % vrbls['dim']
        col = vrbls['mat'][:, coor]
        old_coor = vrbls['beta'][coor]
        new_coor = max(
            old_coor + np.dot(col, vrbls['res']) / vrbls['sq_norms'][coor], 0)
        vrbls['beta'][coor] = new_coor
        diff = old_coor - new_coor
        vrbls['res'] = vrbls['res'] + diff * col
        vrbls['ers'][coor] = abs(diff) * vrbls['norms'][coor]

        vrbls['er'] = np.mean(vrbls['ers'])
        vrbls['errs'].append(vrbls['er'])
        vrbls['ite'] += 1
    return vrbls['beta'], pd.DataFrame(data=vrbls['errs'], columns=['err_rel']), vrbls

def inc_proxes(vrbls, proxes, met):
    '''Update of errors, `prim`, `ite` for each iteration of gradient descent algos.

    Parameters
    ----------
    vrbls : dict
        Current variables.
    proxes : list[function]
        List of proximal operators.
    met : func
        Norm used to measure errors.

    Returns
    -------
    None
        The dictionary `vrbls` is updated inplace.

    '''
    vrbls['eac'] = max(met(prox(vrbls['prim']) - vrbls['prim']) for prox in proxes)
    vrbls['er'] = met(vrbls['prim'] - vrbls['prim_new'])
    vrbls['errs'].append([vrbls['eac'], vrbls['er']])
    vrbls['prim'] = vrbls['prim_new'].copy()
    vrbls['ite'] += 1

def dykstra(vec, projs, met, **maxes):
    '''Project onto an intersection of convex sets,
    defined through their projectors,
    thanks to Dykstra algorithm.

    Parameters
    ----------
    vec : numpy.array
        Input array to project.
    projs : list[function]
        List of projectors onto convex sets.
    met : func
        Norm used to measure errors.
    **maxes : dict, optional
        Dictionary whose keys are among ['er', 'eac', 'ite'].
        These keys stand for: `relative error`, `absolute constraint error`, `iteration`.

    Returns
    -------
    sol : numpy.array
        Projection of `vec` onto the convex sets defined by projs.
    errors : pandas.DataFrame
        Successive maximum absolute projection and relative primal errors for the norm `met`.
    vrbls : dict
        Variables used in the optimization.

    '''
    vrbls = init_algo(None, vec, maxes, {'er': 1e-6, 'eac': 1e-6, 'ite': 3000})
    vrbls.update({
        'prim': vec.copy(),
        'duals': np.zeros((len(projs), vrbls['dim'])),
        'n_proxes': len(projs),
    })
    while keep_iterating(vrbls):
        for idx, proj in enumerate(projs):
            add = vrbls['prim'] + vrbls['duals'][idx]
            vrbls['prim_new'] = proj(add)
            vrbls['duals'][idx] = add - vrbls['prim_new']
        inc_proxes(vrbls, projs, met)
    vrbls['errs'] = pd.DataFrame(data=vrbls['errs'], columns=['err_abs_projs', 'err_rel'])
    return vrbls['prim'], vrbls['errs'], vrbls


def init_gd(vrbls, n_proxes):
    '''Initialize parameters for gradient descent algorithms.

    Parameters
    ----------
    vrbls : dict
        Variables initialized by the function `init_algo`.
    n_proxes : int
        Number of proximal operators.

    Returns
    -------
    None
        The dictionary `vrbls` is further initialized inplace.

    '''
    vrbls['n_proxes'] = n_proxes
    vrbls['step'] = norm(vrbls['mat_sym'])  # Lipschitz constant
    assert vrbls['step'] > 1e-6, (
        f"The norm {vrbls['step']} of the symmetrized `mat` is too small.")
    # gamma = 1 / lip, could be in (0, 2 / lip); lam = 1
    vrbls['mul_mat']      = np.eye(vrbls['dim']) - vrbls['mat_sym'] / vrbls['step']
    vrbls['mul_mat_conj'] = np.eye(vrbls['dim']) + vrbls['mat_sym'] / vrbls['step']
    vrbls['out'] = vrbls['vec'] / vrbls['step']
    # 2 p_r - p - 1 / L * (A p - v) = 2 p_r - vrbls['mul_mat_conj'] p + vrbls['out']
    #                               = vrbls['mul_mat'] p + vrbls['out']   if p_r = p  (1 proj case)
    rng = np.random.default_rng(vrbls['seed'])
    vrbls['prims'] = [rng.standard_normal(vrbls['dim']) for _ in range(n_proxes)]
    vrbls['prim'] = np.array(vrbls['prims']).mean(axis=0)

def pgd(mat, vec, projs, met, maxes, **options):
    '''
    The following problem is solved by the projected gradient descent method:
    min._x   0.5 x.T @ mat @ x - vec.T @ x   s. t.   proj(x) = x for all proj in projs.

    Parameters
    ----------
    mat : numpy.array
        Input real positive semidefinite array, for the objective function.
    vec : numpy.array
        Input array, for the objective function.
    projs : list[function]
        List of projectors onto convex sets (the constraints).
    met : func
        Norm used to measure errors.
    maxes : dict
        Dictionary whose keys are among ['er', 'eac', 'ite']. It can be {}.
        These keys stand for: `relative error`, `absolute constraint error`, `iteration`.
    **options : dict, optional
        The random seed `seed` can be passed.

    Returns
    -------
    sol : numpy.array
        Solution of the optimization problem.
    errors : pandas.DataFrame
        Successive maximum absolute projection and relative primal errors for the norm `met`.
    vrbls : dict
        Variables used in the optimization.

    '''
    assert isinstance(projs, list) and len(projs) >= 1, (
        'The constraints must be passed as a list of functions, '
        + 'there must be at least one constraint.')
    vrbls = init_algo(mat, vec, maxes, {'er': 1e-6, 'eac': 1e-6, 'ite': 2000}, **options)
    init_gd(vrbls, len(projs))

    def update(prim, prim_ref, proj):
        return proj(2 * prim_ref - np.matmul(vrbls['mul_mat_conj'], prim) + vrbls['out'])

    while keep_iterating(vrbls):
        vrbls['prims'] = [
            prim - vrbls['prim'] + update(prim, vrbls['prim'], proj)
            for proj, prim in zip(projs, vrbls['prims'])]
        vrbls['prim_new'] = np.array(vrbls['prims']).mean(axis=0)
        inc_proxes(vrbls, projs, met)
    vrbls['errs'] = pd.DataFrame(data=vrbls['errs'], columns=['err_abs_proj', 'err_rel'])
    return vrbls['prim'], vrbls['errs'], vrbls


def apgd(mat, vec, prox, met, maxes, **options):
    '''
    The following problem is solved by the accelerated proximal gradient descent:
    min._x   0.5 x.T @ mat @ x - vec.T @ x   s. t.   prox(x) = x.

    Parameters
    ----------
    mat : numpy.array
        Input real positive semidefinite array, for the objective function.
    vec : numpy.array
        Input array, for the objective function.
    prox : function
        One proximal operator.
    met : func
        Norm used to measure errors.
    maxes : dict
        Dictionary whose keys are among ['er', 'eac', 'ite']. It can be {}.
        These keys stand for: `relative error`, `absolute constraint error`, `iteration`.
    **options : dict, optional
        The random seed `seed` can be passed.

    Returns
    -------
    sol : numpy.array
        Solution of the optimization problem.
    errors : pandas.DataFrame
        Successive maximum absolute projection and relative primal errors for the norm `met`.
    vrbls : dict
        Variables used in the optimization.

    '''
    vrbls = init_algo(mat, vec, maxes, {'er': 1e-6, 'eac': 1e-6, 'ite': 1000}, **options)
    init_gd(vrbls, 1)
    vrbls['t_old'] = 1

    update = lambda prim_: prox(np.matmul(vrbls['mul_mat'], prim_) + vrbls['out'])

    while keep_iterating(vrbls):
        vrbls['prim_new'] = update(vrbls['prim'])
        vrbls['ttt'] = 0.5 * (1 + np.sqrt(1 + 4 * vrbls['t_old'] ** 2))
        ratio = (vrbls['t_old'] - 1) / vrbls['ttt']
        vrbls['prim_new'] = (1 + ratio) * vrbls['prim_new'] - ratio * vrbls['prim']
        vrbls['t_old'] = vrbls['ttt']
        inc_proxes(vrbls, [prox], met)
    vrbls['errs'] = pd.DataFrame(data=vrbls['errs'], columns=['err_abs_proj', 'err_rel'])
    return prox(vrbls['prim']), vrbls['errs'], vrbls

def form_mat_qp(mat, mat_consts=None):
    '''Form the augmented matrix (lhs) to solve the quadratic problem with linear constraints:
    min._x   0.5 * x.T @ mat @ x   s. t.   mat_consts @ x = 0.

    Parameters
    ----------
    mat : numpy.array
        Input 2d real positive semidefinite array, for the objective function.
    mat_consts : numpy.array or None, optional
        Input 2d array encoding the constraints.

    Returns
    -------
    sol : numpy.array
        Augmented matrix.

    '''
    dim = len(mat)
    if mat_consts is None:  # unconstrained default case
        mat_consts = np.empty((0, dim))
    sdim = dim + len(mat_consts)
    aug_mat = np.zeros((sdim, sdim))
    aug_mat[:dim, :dim] = 0.5 * (mat + mat.T)
    aug_mat[dim:, :dim] = mat_consts
    aug_mat[:dim, dim:] = mat_consts.T
    return aug_mat

def form_vec_qp(vec, vec_consts=np.empty(0)):
    '''Form the augmented vector (rhs) to solve the quadratic problem with linear constraints:
    min._x   0.5 * x.T @ mat @ x - vec.T @ x   s. t.   mat_consts @ x = vec_consts,
    where `mat`, `mat_consts` are some matrices not passed here.

    Parameters
    ----------
    vec : numpy.array
        Input 1d array.
    vec_consts : numpy.array, optional
        Input 1d array encoding the constraints.

    Returns
    -------
    sol : numpy.array
        Concatenated arrays.

    '''
    return np.concatenate([vec, vec_consts])

def update_rho_admm_cache(vrbls):
    '''Update the penalization parameter in `admm`.
    Eventually updates the values for `rho` and `dual`, `inv_reg`.

    This heuristic was first published in: He, B.S., Yang, H. & Wang, S.L.
    Alternating Direction Method with Self-Adaptive Penalty Parameters
    for Monotone Variational Inequalities.
    Journal of Optimization Theory and Applications 106, 337–356 (2000).

    Parameters
    ----------
    vrbls : dict
        Dictionary of current variables.

    Returns
    -------
    None

    '''
    ratio_errs = vrbls['ep'] / vrbls['ed']  # ratio errors primal / dual
    if ratio_errs > vrbls['ratio_max']:
        vrbls['rho'] *= vrbls['tau']
        vrbls['dual'] = vrbls['dual'] / vrbls['tau']
        vrbls['inv_reg'] = inv(form_mat_qp(
            vrbls['mat'] + vrbls['rho'] * np.eye(vrbls['dim']), vrbls['mat_consts']))
    elif ratio_errs < 1 / vrbls['ratio_max']:
        vrbls['rho'] /= vrbls['tau']
        vrbls['dual'] = vrbls['dual'] * vrbls['tau']
        vrbls['inv_reg'] = inv(form_mat_qp(
            vrbls['mat'] + vrbls['rho'] * np.eye(vrbls['dim']), vrbls['mat_consts']))

def inc_admm(vrbls, met):
    '''Update of errors, `prim_2`, `ite` for each iteration of admm's.

    Parameters
    ----------
    vrbls : dict
        Current variables.
    met : func
        Norm used to measure errors.

    Returns
    -------
    None
        The dictionary `vrbls` is updated inplace.

    '''
    vrbls['ep'] = met(vrbls['prim_1'] - vrbls['prim_2_new'])  # primal error
    vrbls['ed'] = vrbls['rho'] * met(vrbls['prim_2'] - vrbls['prim_2_new'])  # dual error
    vrbls['errs'].append([vrbls['ep'], vrbls['ed'], vrbls['rho']])
    vrbls['prim_2'] = vrbls['prim_2_new'].copy()
    vrbls['ite'] += 1

# def admm(mat, vec, proj, met, maxes, **options):
#     '''The following problem is solved by the admm:
#     min._x   0.5 x.T @ mat @ x - vec.T @ x   s. t.   proj(x) = x.
#     Extra linear constraints:  mat_consts @ x = vec_consts,
#     can be passed through the dictionary `options`.

#     Parameters
#     ----------
#     mat : array
#         Input real positive semidefinite array, for the objective function.
#     vec : array
#         Input array, for the objective function.
#     proj : function
#         Projection onto the constraints.
#     met : func
#         Norm used to measure errors.
#     maxes : dict
#         Dictionary whose keys are among ['ep', 'ed', 'ite']. It can be {}.
#         These keys stand for: `primal error`, `dual error`, `iteration`.
#     **options : dict, optional
#         Arguments `mat_consts` and `vec_consts` can be passed.
#         Extra arguments `ratio_max`, `tau` for the function `update_rho_admm_cache` can be passed.
#         The random seed `seed` can be passed.

#     Returns
#     -------
#     sol : array
#         Solution of the optimization problem.
#     errors : pandas.DataFrame
#         Successive primal and dual errors for the norm `met` and successive values of `rho`.
#     vrbls : dict
#         Variables used in the optimization.

#     '''
#     vrbls = init_algo(mat, vec, maxes, {'ep': 1e-5, 'ed': 1e-5, 'ite': 10000})
#     vrbls.update({  # default parameters and admm initializations
#         'tau': 2,
#         'ratio_max': 10,
#         'mat_consts': None,
#         'vec_consts': np.empty(0),
#         'rho': 1,
#         'prim_2': np.random.default_rng(vrbls['seed']).standard_normal(vrbls['dim']),
#         'dual': np.zeros(vrbls['dim']),
#     })
#     vrbls['inv_reg'] = inv(form_mat_qp(  # cached inverse of the matrix used in prox computation
#         vrbls['mat'] + vrbls['rho'] * np.eye(vrbls['dim']), vrbls['mat_consts']))
#     while keep_iterating(vrbls):
#         rhs = form_vec_qp(
#             vrbls['rho'] * (vrbls['prim_2'] - vrbls['dual']),
#             vrbls['vec_consts'])
#         vrbls['prim_1'] = np.matmul(vrbls['inv_reg'], rhs)[:vrbls['dim']]
#         vrbls['prim_2_new'] = proj(vrbls['prim_1'] + vrbls['dual'])
#         vrbls['dual'] = vrbls['dual'] + vrbls['prim_1'] - vrbls['prim_2_new']
#         update_rho_admm_cache(vrbls)
#         inc_admm(vrbls, met)
#     vrbls['errs'] = pd.DataFrame(data=vrbls['errs'], columns=['err_prim', 'err_dual', 'rho'])
#     return vrbls['prim_2'], vrbls['errs'], vrbls
#     # 0.5 * (vrbls['prim_1'] + vrbls['prim_2'])

def admm_consensus(mat, vec, projs, met, maxes, **options):
    '''The following problem is solved by the admm:
    min._x   0.5 x.T @ mat @ x - vec.T @ x   s. t.   proj(x) = x.
    Extra linear constraints:  mat_consts @ x = vec_consts,
    can be passed through the dictionary `options`.

    Parameters
    ----------
    mat : numpy.array
        Input real positive semidefinite array, for the objective function.
    vec : numpy.array
        Input array, for the objective function.
    projs : list[function]
        List of projectors onto convex sets (the constraints).
    met : func
        Norm used to measure errors.
    maxes : dict
        Dictionary whose keys are among ['ep', 'ed', 'eac', 'ite']. It can be {}.
        These keys stand for: `primal error`, `dual error`, `absolute constraints error` and
        `iteration`.
    **options : dict, optional
        Arguments `mat_consts` and `vec_consts` can be passed.
        Extra arguments `ratio_max`, `tau` for the function `update_rho_admm_cache` can be passed.
        The random seed `seed` can be passed.

    Returns
    -------
    sol : numpy.array
        Solution of the optimization problem.
    errors : pandas.DataFrame
        Successive maximal primal, dual and maximal constraints errors
        for the norm `met` and successive values of the penalization parameter `rho`.
    vrbls : dict
        Variables used in the optimization.

    '''
    vrbls = init_algo(mat, vec, maxes, {
        'ep': 1e-5, 'ed': 1e-5, 'eac': 1e-5, 'ite': 10000}, **options)
    vrbls.update({  # default parameters and admm initializations
        'tau': 2,
        'ratio_max': 10,
        'mat_consts': None,
        'vec_consts': np.empty(0),
        'rho': 1,
        'prim_2': np.random.default_rng(vrbls['seed']).standard_normal(vrbls['dim']),
        'dual': np.zeros(vrbls['dim']),
        'n_proxes': len(projs),
        'duals': [np.zeros(vrbls['dim']) for _ in range(len(projs))]
    })
    vrbls['inv_reg'] = inv(form_mat_qp(  # cached inverse of the matrix used in prox computation
        vrbls['mat'] + vrbls['rho'] * np.eye(vrbls['dim']), vrbls['mat_consts']))

    while keep_iterating(vrbls):
        vrbls['prims_1'] = [
            proj(vrbls['prim_2'] + dual) for proj, dual in zip(projs, vrbls['duals'])]
        vrbls['prim_1'] = np.array(vrbls['prims_1']).mean(axis=0)
        rhs = form_vec_qp(
            vrbls['n_proxes'] * vrbls['rho'] * (vrbls['prim_1'] - vrbls['dual']),
            vrbls['vec_consts'])
        vrbls['prim_2_new'] = np.matmul(vrbls['inv_reg'], rhs)[:vrbls['dim']]
        vrbls['duals'] = [
            dual + prim_1 - vrbls['prim_2_new']
            for prim_1, dual in zip(vrbls['prims_1'], vrbls['duals'])]
        vrbls['dual'] = np.array(vrbls['duals']).mean(axis=0)
        update_rho_admm_cache(vrbls)

        # primal, dual and constraints errors
        vrbls['ep'] = max(met(prim_1 - vrbls['prim_2_new']) for prim_1 in vrbls['prims_1'])
        vrbls['ed'] = vrbls['n_proxes'] ** .5 * vrbls['rho'] * met(
            vrbls['prim_2'] - vrbls['prim_2_new'])
        vrbls['eac'] = max(met(proj(vrbls['prim_2_new']) - vrbls['prim_2_new']) for proj in projs)
        vrbls['errs'].append([vrbls['ep'], vrbls['ed'], vrbls['eac'], vrbls['rho']])
        vrbls['prim_2'] = vrbls['prim_2_new'].copy()
        vrbls['ite'] += 1
    vrbls['errs'] = pd.DataFrame(data=vrbls['errs'], columns=[
        'err_prim', 'err_dual', 'err_abs_proj', 'rho'])
    return vrbls['prim_2'], vrbls['errs'], vrbls
