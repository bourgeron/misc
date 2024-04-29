#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.linalg import norm
from scipy.sparse import diags, csr_matrix
from display_functions import plot_orders, save_anim


def build_scheme_matrix(name, alpha, n_steps_space):
    if name == 'centered_explicit':
        half_alpha = 0.5 * alpha
        diag1 = half_alpha * np.ones(n_steps_space - 1)
        tridiag = diags(
            [diag1, np.ones(n_steps_space), -diag1],
            [-1, 0, 1], format='lil')
        tridiag[0, n_steps_space - 1] = half_alpha
        tridiag[n_steps_space - 1, 0] = -half_alpha
    elif name == 'backward':  # 'upwind' as c > 0
        tridiag = diags(
            [alpha * np.ones(n_steps_space - 1),
             (1 - alpha) * np.ones(n_steps_space)],
            [-1, 0], format='lil')
        tridiag[0, n_steps_space - 1] = alpha
    elif name == 'Lax–Friedrichs':
        alpha_p1 = 0.5 * (1 + alpha)
        alpha_m1 = 0.5 * (1 - alpha)
        tridiag = diags(
            [alpha_p1 * np.ones(n_steps_space - 1),
             alpha_m1 * np.ones(n_steps_space - 1)],
            [-1, 1], format='lil')
        tridiag[0, n_steps_space - 1] = alpha_p1
        tridiag[n_steps_space - 1, 0] = alpha_m1
    elif name == 'Lax–Wendroff':
        alpha_p1 = 0.5 * alpha * (alpha + 1)
        alpha_m1 = 0.5 * alpha * (alpha - 1)
        tridiag = diags(
            [alpha_p1 * np.ones(n_steps_space - 1),
             (1 - alpha ** 2) * np.ones(n_steps_space),
             alpha_m1 * np.ones(n_steps_space - 1)],
            [-1, 0, 1], format='lil')
        tridiag[0, n_steps_space - 1] = alpha_p1
        tridiag[n_steps_space - 1, 0] = alpha_m1
    return tridiag.tocsr()

implemented_schemes = [
    'centered_explicit', 'backward',
    'Lax–Friedrichs', 'Lax–Wendroff']

def compute_init_vals(n_steps_space, params):
    speed, end_time, alpha, fun_init = params
    step_space = 1 / n_steps_space
    n_steps_time = int(end_time * speed / (alpha * step_space))
    step_time = end_time / n_steps_time
    alpha = speed * step_time / step_space
    grid_space = np.linspace(0, 1 - step_space, n_steps_space)
    vals = [None] * (n_steps_time + 1)
    val_init = fun_init(grid_space)
    times = np.linspace(0, end_time, n_steps_time + 1)
    return (
        grid_space, vals, val_init, alpha,
        times, n_steps_time, step_time)

def scheme(name, n_steps_space, params):
    assert name in implemented_schemes, 'not implemented'
    (
        grid_space, vals, val_init,
        alpha, times, n_steps_time, step_time
    ) = compute_init_vals(n_steps_space, params)    

    mat = build_scheme_matrix(name, alpha, n_steps_space)
    vals[0] = csr_matrix(val_init.reshape(-1, 1))
    for step_time in range(0, n_steps_time):
        vals[step_time + 1] = mat * vals[step_time]
    vals = [val.toarray() for val in vals]

    return grid_space, vals, val_init, alpha, times


steep_fun = (
    lambda grid: 1 * (grid >= 1 / 3) * (grid < 2 / 3))
# steep_fun = (lambda grid: np.abs(np.sin(np.pi * grid)) ** 10)
# steep_fun = (lambda grid: np.sin(10 * np.pi * grid))
params_common = (1, 1, 0.3, steep_fun)

for method in implemented_schemes:
    save_anim(scheme, method, 50, params_common)


n_steps_grid = 2 ** np.arange(9, 13)

errs = dict()
for method in implemented_schemes[1:]:
    errs[method] = list()
    for n_steps in n_steps_grid:
        grid, values, true_val, _, _ = scheme(
            method, n_steps, params_common)
        dx = grid[1] - grid[0]
        diff = values[-1] - true_val.reshape(-1, 1)
        errs[method].append(norm(diff, 2) * dx ** 0.5)
        
plot_orders(n_steps_grid, errs)
