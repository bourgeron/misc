from functools import reduce
from scipy.linalg import solve_triangular
import numpy as np
from numpy.linalg import solve, inv, cholesky, pinv, LinAlgError, norm
from optim_algos import (
    form_mat_qp, form_vec_qp,
    bisection_increasing, init_prox_lp, prox_lp_1d_newton_raphson)


## projectors

def proj_affine_hyperplane(vec, vec_or, intercept):
    '''Project `vec` onto the affine hyperplane: {x: vec_or.T x = intercept}.

    Parameters
    ----------
    vec : numpy.array
        Input 1d array to project.
    vec_or : numpy.array
        Orientation vector (1d array) defining the hyperplane to project on.
    intercept : float
        Intercept defining the hyperplane to project on.

    Returns
    -------
    sol : numpy.array
        Projection of the vector `vec`.

    '''
    norm_sqr = np.dot(vec_or, vec_or)
    assert norm_sqr > 1e-6, (
        f'The orientation vector must be long: {norm_sqr * .5} is not.')
    signed_dist = (np.dot(vec_or, vec) - intercept) / norm_sqr
    return vec - signed_dist * vec_or

def proj_kernel(vec, mat, vec_consts=None):
    '''Project `vec` onto the affine kernel of `mat`: {x: mat @ x = vec_consts},
    the vector `vec_consts` defaults to 0.
    https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Projectors
    When `mat` is squared, it is equivalent to:
    solve_qp(identity, vec, mat, vec_consts) but 50x faster.

    Parameters
    ----------
    vec : numpy.array
        Input 1d array to project.
    mat : numpy.array
        Two-d array which kernel is used for projection.
    vec_consts : numpy.array or None, optional
        One-d array used as intercept.

    Returns
    -------
    sol : numpy.array
        Projection of `vec`.

    '''
    pinv_mat = pinv(mat)
    proj_ker = vec - reduce(np.matmul, [pinv_mat, mat, vec])
    return proj_ker + 0 if vec_consts is None else np.matmul(pinv_mat, vec_consts)

def proj_canonical_hyperplane(vec, idx):
    '''Set the `idx`-th coordinate of `vec` to 0.

    Parameters
    ----------
    vec : numpy.array
        Vector to be projected.
    idx : int
        Coordinate.

    Returns
    -------
    sol : numpy.array
        Projection of `vec`.

    '''
    vec[idx] = 0
    return vec

def proj_simplex(vec, rad=1):
    '''Project `vec` onto the simplex: {x: x >=0 and sum(x) = rad},
    the index used to translate `vec` is found by sorting `vec` entirely:
    Held, M., Wolfe, P. and Crowder, H.P., 1974. Validation of subgradient optimization.
    Mathematical programming, 6(1), pp.62-88.

    Parameters
    ----------
    vec : numpy.array
        Input 1d array to project.
    rad : float, optional
        Sum of the elements of the simplex (positive float).

    Returns
    -------
    sol : numpy.array
        Projection of the vector `vec`.

    '''
    assert rad > 0
    assert all((isinstance(vec, np.ndarray), vec.ndim == 1, len(vec) >= 1)), (
        'The imput vector `vec` must be a 1-d non empty numpy array.')
    sorted_vec = np.sort(vec)[::-1]
    cummeans = 1 / np.arange(1, len(vec) + 1) * (np.cumsum(sorted_vec) - rad)
    rho = max(np.where(sorted_vec > cummeans)[0])
    return np.maximum(vec - cummeans[rho], 0)

def proj_simplex_quick(vec, rad=1):
    '''Project `vec` onto the simplex: {x: x >=0 and sum(x) = rad},
    the index used to translate `vec` is found by an algorithm similar to quicksort:
    Kiwiel, K.C., 2008.
    Breakpoint searching algorithms for the continuous quadratic knapsack problem.
    Mathematical Programming, 112(2), pp.473-491.

    Parameters
    ----------
    vec : numpy.array
        Input 1d array to project.
    rad : float, optional
        Sum of the elements of the simplex (positive float).

    Returns
    -------
    sol : numpy.array
        Projection of the vector `vec`.

    '''
    assert rad > 0
    assert all((isinstance(vec, np.ndarray), vec.ndim == 1, len(vec) >= 1)), (
        'The imput vector `vec` must be a 1-d non empty numpy array.')
    vals = vec.copy()
    length = 0
    som = -rad
    while vals.size > 0:
        val = np.random.choice(vals)  # choose the median...
        high = vals[vals > val]
        low = vals[vals < val]
        n_eq = len(vals) - len(high) - len(low)
        som_ = som + n_eq * val + high.sum()
        length_ = length + n_eq + len(high)
        if som_ / length_ < val:
            vals = low.copy()
            length = length_
            som = som_
        else:
            vals = high.copy()
    return np.maximum(vec - som / length, 0)

def proj_simplex_active_set(vec, rad=1):
    '''Project `vec` onto the simplex: {x: x >=0 and sum(x) = rad},
    the index used to translate `vec` is found by an active set method:
    Michelot, C., 1986.
    A finite algorithm for finding the projection of a point onto the canonical simplex of R^n.
    Journal of Optimization Theory and Applications, 50(1), pp.195-200.

    Parameters
    ----------
    vec : numpy.array
        Input 1d array to project.
    rad : float, optional
        Sum of the elements of the simplex (positive float).

    Returns
    -------
    sol : numpy.array
        Projection of the vector `vec`.

    '''
    assert rad > 0
    assert all((isinstance(vec, np.ndarray), vec.ndim == 1, len(vec) >= 1)), (
        'The imput vector `vec` must be a 1-d non empty numpy array.')
    vals = vec.copy()
    length_old = len(vals)
    mean = (sum(vals) - rad) / length_old
    lengths_diff = 1
    while lengths_diff >= 1:
        vals = vals[vals > mean]
        length = len(vals)
        mean = (sum(vals) - rad) / length
        lengths_diff = abs(length - length_old)
        length_old = length
    return np.maximum(vec - mean, 0)

def proj_simplex_condat(vec, rad=1):
    '''Project `vec` onto the simplex: {x: x >=0 and sum(x) = rad}:
    Condat, L., 2016.
    Fast projection onto the Simplex and the l1 Ball.
    Mathematical Programming, 158(1), pp.575-585.

    Parameters
    ----------
    vec : numpy.array
        Input 1d array to project.
    rad : float, optional
        Sum of the elements of the simplex (positive float).

    Returns
    -------
    sol : numpy.array
        Projection of the vector `vec`.

    '''
    assert rad > 0
    assert all((isinstance(vec, np.ndarray), vec.ndim == 1, len(vec) >= 1)), (
        'The imput vector `vec` must be a 1-d non empty numpy array.')
    largest = [vec[0]]
    reservoir = []
    mean = largest[0] - rad
    for val in vec[1:]:
        if val > mean:
            mean = mean + (val - mean) / (len(largest) + 1)
            if mean > val - rad:
                largest.append(val)
            else:
                reservoir.extend(largest)
                largest = [val]
                mean = val - rad
    if reservoir:
        for val in reservoir:
            if val > mean:
                largest.append(val)
                mean = mean + (val - mean) / len(largest)
    length_old = len(largest)
    lengths_diff = 1
    while lengths_diff >= 1:
        for ind, val in enumerate(largest):
            if val <= mean:
                del largest[ind]
                mean = mean + (mean - val) / len(largest)
        length = len(largest)
        lengths_diff = abs(length - length_old)
        length_old = length
    return np.maximum(vec - mean, 0)

def proj_l1_ball(vec, rad=1):
    '''Project `vec` onto the l1 ball: {x: ||x||_1 <= rad}.
    It relies on `proj_simplex_quick`.

    Parameters
    ----------
    vec : numpy.array
        Input 1d array to project.
    rad : float, optional
        Radius of the ball (positive float).

    Returns
    -------
    sol : numpy.array
        Projection of the vector `vec`.

    '''
    assert rad > 0
    abs_vec = np.abs(vec)
    triv = abs_vec.sum() <= rad
    return vec if triv else np.sign(vec) * proj_simplex_quick(abs_vec, rad)

def proj_l2_ball(vec, rad=1):
    '''Project `vec` onto the l2 ball: {x: ||x||_2 <= rad}.

    Parameters
    ----------
    vec : numpy.array
        Input 1d array to project.
    rad : float, optional
        Radius of the ball (positive float).

    Returns
    -------
    sol : numpy.array
        Projection of the vector `vec`.

    '''
    assert rad > 0
    norm_rad = norm(vec) / rad
    return vec if norm_rad <= 1 else vec / norm_rad

def proj_box(vec, lwb, upb):  # proj_linfty
    '''Project `vec` onto the box `[lwb, upb]`.

    Parameters
    ----------
    vec : numpy.array
        Input 1d array to project.
    lwb, upb : array_like or None
        Lower and upper bounds.

    Returns
    -------
    sol : numpy.array
        Clipped array `vec`.

    '''
#     assert (lwb <= vec).all() and (vec <= upb).all()
    return vec.clip(lwb, upb)

def prox_conjugate(vec, prox):
    '''Proximal operator of the convex conjugate, Moreau  decomposition formula.

    Parameters
    ----------
    vec : numpy.array
        Input 1d array to project.
    prox : function
        Proximal operator of some function f.

    Returns
    -------
    sol : numpy.array
        Value of the proximal operator of f* at vec.

    '''
    return vec - prox(vec)


## linearly constrained quadratic problems

def solve_qp(mat, vec, mat_consts=None, vec_consts=np.empty(0)):
    '''Solve the quadratic problem with linear constraints:
    min._x   0.5 * x.T @ mat @ x - vec.T @ x   s. t.   mat_consts @ x = vec_consts.

    Parameters
    ----------
    mat : numpy.array
        Input real positive semidefinite array, for the objective function.
    vec : numpy.array
        Input 1d array.
    mat_consts : numpy.array or None, optional
        Input 2d array encoding the constraints.
    vec_consts : numpy.array, optional
        Input 1d array encoding the constraints.

    Returns
    -------
    sol : numpy.array
        Solution to the linearly constrained quadratic problem.

    '''
    aug_mat = form_mat_qp(mat, mat_consts)
    aug_vec = form_vec_qp(vec, vec_consts)
    try:
        sol = solve(aug_mat, aug_vec)
    except LinAlgError as err:
        if 'Singular matrix' in str(err):
            sol = np.matmul(pinv(aug_mat), aug_vec)
    return sol[:len(mat)]

def qp_chol(mat, vec, mat_consts, vec_consts):
    '''Same as `solve_qp` but uses Cholesky decompositions and the Schur complement of `mat`.
    Sadly, it is slower than `solve_qp`.

    Solve the quadratic problem with linear constraints:
    min._x   0.5 * x.T @ mat @ x - vec.T @ x   s. t.   mat_consts @ x = vec_consts.

    Parameters
    ----------
    mat : numpy.array
        Input real positive semidefinite array, for the objective function.
    vec : numpy.array
        Input 1d array.
    mat_consts : numpy.array or None, optional
        Input 2d array encoding the constraints.
    vec_consts : numpy.array, optional
        Input 1d array encoding the constraints.

    Returns
    -------
    sol : numpy.array
        Solution to the linearly constrained quadratic problem.

    '''
    mat_sym = 0.5 * (mat + mat.T)
    mat_consts_inv_mat = np.matmul(mat_consts, inv(mat_sym))
    schur = np.matmul(mat_consts_inv_mat, mat_consts.T)
    p_trig = {'overwrite_b': True, 'check_finite': False}

    chol_schur = cholesky(schur)
    rhs1 = vec_consts - np.matmul(mat_consts_inv_mat, vec)
    muu = solve_triangular(chol_schur, rhs1, lower=True, **p_trig)
    lam = solve_triangular(chol_schur.T, muu, lower=False, **p_trig)

    chol_mat = cholesky(mat)
    rhs2 = vec + np.matmul(mat_consts.T, lam)
    step = solve_triangular(chol_mat, rhs2, lower=True, **p_trig)
    return solve_triangular(chol_mat.T, step, lower=False, **p_trig)

# def solve_qp_1d_overdetermined(mat, vec, mat_consts, vec_consts, vec_ref):
#     aug_mat = form_mat_qp(mat, mat_consts)
#     aug_vec = form_vec_qp(vec, vec_consts)
#     t_term = np.matmul(aug_mat, vec_ref)
#     sol = np.matmul(pinv(aug_mat), aug_vec - t_term) + vec_ref
#     return sol[:len(mat)]

# def qp_opt(mat, vec, mat_consts, vec_consts):
#     return (
#         qp_chol(mat, vec, mat_consts, vec_consts)
#         if matrix_rank(mat_consts) == mat_consts.shape[0]
#         else solve_qp(mat, vec, mat_consts, vec_consts))

def prox_qp(point, mat, vec, mat_consts, vec_consts, lam):
    '''Proximal operator of the following function at `point`:
    lam * (0.5 * x.T @ mat @ x - vec.T @ x)
    with domain: {x: mat_consts @ x = vec_consts}.

    Parameters
    ----------
    point : numpy.array
        Point at which the proximal operator is computed.
    mat : numpy.array
        Input real positive semidefinite array, for the objective function.
    vec : numpy.array
        Input 1d array.
    mat_consts : numpy.array or None
        Input 2d array encoding the constraints.
    vec_consts : numpy.array
        Input 1d array encoding the constraints.
    lam : float
        Non-negative parameter.

    Returns
    -------
    sol : numpy.array
        Value of the proximal operator at `point`.

    '''
    assert lam >= 0
    return solve_qp(
        lam * mat + np.identity(len(mat)),
        lam * vec + point,
        mat_consts,
        vec_consts)


## proxes lp norms

def prox_l1(point, vec_a, lam):
    '''Proximal operator of the following function at `point`: x |-> lam * dot(vec_a, abs(x)).
    Same as `prox_lp` with expo = 1. It relies on an explicit formula.

    Parameters
    ----------
    point : numpy.array
        One-d array, point at which the value is sought.
    vec_a : numpy.array
        Non-negative 1-d array.
    lam : float
        Non-negative parameter.

    Returns
    -------
    sol : numpy.array
        Value of the proximal operator at `point`.

    '''
    assert (vec_a >= 0).all() and lam >= 0
    return np.minimum(point + lam * vec_a, 0) + np.maximum(point - lam * vec_a, 0)

# def prox_l1_gen(point, vec_a, vec_t, lam):
#     res = vec_t + prox_l1(point - vec_t, vec_a, lam)
#     return res

def prox_l1dot5(point, vec_b, lam):
    '''Proximal operator of the following function at point:
    x |-> lam * 1 / 1.5 * dot(vec_b, abs(x) ^ 1.5).
    Same as `prox_lp` with expo = 1.5. It relies on an explicit formula.

    Parameters
    ----------
    point : numpy.array
        One-d array, point at which the value is sought.
    vec_b : numpy.array
        Non-negative 1-d array.
    lam : float
        Non-negative parameter.

    Returns
    -------
    sol : numpy.array
        Proximal operator at `point`.
    '''
    assert (vec_b >= 0).all() and lam >= 0
    term_0 = 0.5 * lam * vec_b
    term_1 = term_0 ** 2
    term_2 = term_1 + np.abs(point)
    return np.sign(point) * (term_1 + term_2 - 2 * term_0 * np.sqrt(term_2))

def prox_l2(point, vec_c, lam):
    '''Proximal operator of the following function at point:
    x |-> lam * 1 / 2 * dot(vec_c, abs(x) ^ 2).
    Same as `prox_lp` with expo = 2.

    Parameters
    ----------
    point : numpy.array
        One-d array, point at which the value is sought.
    vec_c : numpy.array
        Non-negative 1-d array.
    lam : float
        Non-negative parameter.

    Returns
    -------
    sol : numpy.array
        Proximal operator at `point`.
    '''
    assert (vec_c >= 0).all() and lam >= 0
    return point / (1 + lam * vec_c)

def prox_negative_part(point, vec_s, lam):
    '''Proximal operator of the following function at `point`: x |-> lam * dot(vec_s, x)_.

    Parameters
    ----------
    point : numpy.array
        One-d array, point at which the value is sought.
    vec_s : numpy.array
        Non-negative 1-d array.
    lam : float
        Non-negative parameter.

    Returns
    -------
    sol : numpy.array
        Value of the proximal operator at `point`.

    '''
    assert (vec_s >= 0).all() and lam >= 0
    return np.minimum(point + lam * vec_s, 0) + np.maximum(point, 0)

def d_n_p(point, expo, cst, lam):
    '''Value of: point + the derivative of the following function at point:
    w |--> lam * cst * 1 / expo * abs(w) ** expo + 0.5 * (w - point) ** 2.

    Parameters
    ----------
    point : numpy.array
        One-d array, point at which the value is sought.
    expo : float
        Exponent: expo > 1.
    cst, lam : float
        Non-negative parameters. As the functional, the result depends only on their product.
        The constant `cst` is here for easy generalization to the multi-d setting.

    Returns
    -------
    sol : numpy.array
        Value of the quantity at `point`.

    '''
    assert cst >= 0 and lam >= 0 and expo > 1
    return lam * cst * np.sign(point) * np.abs(point) ** (expo - 1) + point

def d2_n_p(point, expo, cst, lam):
    '''Second derivative of the following function at point:
    w |--> lam * cst * 1 / expo * abs(w) ** expo + 0.5 * (w - point) ** 2.

    Parameters
    ----------
    point : numpy.array
        One-d array, point at which the value is sought.
    expo : float
        Exponent: expo > 1.
    cst, lam : float
        Non-negative parameters. As the functional, the result depends only on their product.
        The constant `cst` is here for easy generalization to the multi-d setting.

    Returns
    -------
    sol : numpy.array
        Value of the second derivative at `point`.

    '''
    assert cst >= 0 and lam >= 0 and expo > 1
    return lam * cst * (expo - 1) * np.abs(point) ** (expo - 2) + 1

def prox_lp_1d_bisection(point, expo, cst, lam, point_0=None):
    '''Proximal operator of the following function at point:
    x |-> lam * cst * 1 / expo * abs(x) ^ expo, in dimension 1.
    Computed by a bisection method.

    Parameters
    ----------
    point : numpy.array
        One-d array, point at which the value is sought.
    expo : float
        Exponent: expo >= 1.
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
    assert cst >= 0 and lam >= 0 and expo > 1
    if point_0 is None:
        point_0 = init_prox_lp(point, expo, cst, lam)
    res = bisection_increasing(
        lambda val: d_n_p(val, expo, cst, lam),
        point, min(0, point_0), max(0, point_0))
    return res

def prox_lp_1d(point, expo, cst, lam):
    '''Proximal operator of the following function at point:
    x |-> lam * cst * 1 / expo * abs(x) ^ expo, in dimension 1.
    It uses a Newton-Raphson method, then,
    a bisection method if it has not converged quickly.

    Parameters
    ----------
    point : numpy.array
        One-d array, point at which the value is sought.
    expo : float
        Exponent: expo > 1.
    cst, lam : float
        Non-negative parameters. As the functional, the result depends only on their product.
        The constant `cst` is here for easy generalization to the multi-d setting.

    Returns
    -------
    sol : numpy.array
        Proximal operator at `point`.
    '''
    assert cst >= 0 and lam >= 0 and expo > 1
    point_0 = init_prox_lp(point, expo, cst, lam)
    if expo >= 1.5:
        sol_, ite = prox_lp_1d_newton_raphson(point, expo, cst, lam, point_0)
        sol = sol_ if ite < 20 else prox_lp_1d_bisection(point, expo, cst, lam, point_0)
    else:
        sol = prox_lp_1d_bisection(point, expo, cst, lam, point_0)
    return sol

prox_lp_nd = np.vectorize(prox_lp_1d)  # (point, expo, cst, lam) --> (point, expo, vec_b, lam)

def prox_lp(point, expo, vec_b, lam):
    '''Proximal operator of the following function at point:
    x |-> lam * 1 / expo * dot(vec_b, abs(x) ^ expo).
    For expo in [1, 1.5, 2], specific algorithms are used,
    otherwise `prox_lp_nd` is used.

    Parameters
    ----------
    point : numpy.array
        One-d array, point at which the value is sought.
    expo : float
        Exponent: expo > 1.
    vec_b : numpy.array
        Input 1-d array.
    lam : float
        Non-negative parameter.

    Returns
    -------
    sol : numpy.array
        Proximal operator at `point`.
    '''
    assert (vec_b >= 0).all() and lam >= 0 and expo > 1
    proxes_lp = {1: prox_l1, 1.5: prox_l1dot5, 2: prox_l2}
    if expo in proxes_lp:
        sol = proxes_lp[expo](point, vec_b, lam)
    else:
        sol = prox_lp_nd(point, expo, vec_b, lam)
    return sol

def prox_mixed_lp(point, params, lam):
    '''Proximal operator of the following function at point:
    x |-> lam * (
        dot(vec_a, abs(x)) + 1 / expo * dot(vec_b, abs(x) ** expo)
        + 0.5 * dot(vec_c, x ** 2) + dot(vec_s, x_)),
    where: expo, vec_a, vec_b, vec_c, vec_s = params.
    This proximal operator generalizes: `prox_lp`, `prox_negative_part`:
    `prox_l1` for params = (wtv, vec_a, 0, 0, 0), it does not depend on wtv,
    `prox_lp` for params = (expo, 0, vec_b, 0, 0),
    `prox_l2` for params = (wtv, 0, 0, vec_c, 0), it does not depend on wtv,
    `prox_negative_part` for params = (wtv, 0, 0, 0, vec_s), it does not depend on wtv.

    Parameters
    ----------
    point : numpy.array
        One-d array, point at which the value is sought.
    params : tuple or list
        Parameters expo, vec_a, vec_b, vec_c, vec_s.
    lam : float
        Non-negative parameter.

    Returns
    -------
    sol : numpy.array
        Proximal operator at `point`.
    '''
    assert lam >= 0
    expo, vec_a, vec_b, vec_c, vec_s = params
    lamc = lam / (lam * vec_c + 1)
    pointc = point / (lam * vec_c + 1)
    left = prox_lp(pointc - lamc * vec_a, expo, vec_b, lamc)
    right = prox_lp(pointc + lamc * (vec_a + vec_s), expo, vec_b, lamc)
    return np.minimum(left, 0) + np.maximum(right, 0)

def prox_mixed_lp_box(point, params, lam):
    '''Proximal operator of the following function at point:
    x |-> lam * (
        dot(vec_a, abs(x)) + 1 / expo * dot(vec_b, abs(x) ** expo)
        + 0.5 * dot(vec_c, x ** 2) + dot(vec_s, x_)),
    with domain: {x: v_min <= x <= v_max}.
    where: expo, vec_a, vec_b, vec_c, vec_s, v_min, v_max = params.

    Parameters
    ----------
    point : numpy.array
        One-d array, point at which the value is sought.
    params : tuple or list
        Parameters expo, vec_a, vec_b, vec_c, vec_s, v_min, v_max.
    lam : float
        Non-negative parameter.

    Returns
    -------
    sol : numpy.array
        Proximal operator at `point`.
    '''
    assert lam >= 0
    prox = prox_mixed_lp(point, params[:5], lam)
    return proj_box(prox, *(params[5:]))
