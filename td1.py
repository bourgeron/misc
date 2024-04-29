#!/usr/bin/env python
# coding: utf-8

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from display_functions import plot_orders

def build_laplacian_neumann(n_steps):
    step = 1 / n_steps
    pow_step = n_steps ** 2
    diag0 = 2 * np.ones(n_steps)
    diag0[-1] = step
    diag_1 = -np.ones(n_steps - 1)
    diag_1[-1] = -step    
    diag1 = -np.ones(n_steps - 1)
    tridiag = pow_step * diags(
        [diag_1, diag0, diag1], [-1, 0, 1], format='lil')
    return tridiag.tocsr()

def build_laplacian_dirichlet(n_steps):
    pow_step = n_steps ** 2
    diag0 = 2 * np.ones(n_steps - 1)
    diag1 = -np.ones(n_steps - 2)
    tridiag = pow_step * diags(
        [diag1, diag0, diag1], [-1, 0, 1], format='lil')
    return tridiag.tocsr()

def dirichlet(params, n_steps, true_sol=None, plot=False):
    fun_c, fun_f, l_val, r_val = params
    step = 1 / n_steps
    grid_interior = np.linspace(step, 1 - step, n_steps - 1)

    rhs = fun_f(grid_interior)
    rhs[0] += l_val * step ** -2
    rhs[n_steps - 2] += r_val * step ** -2

    sol = np.zeros(n_steps + 1)
    sol[0] = l_val
    sol[-1] = r_val   
    sol[1:n_steps] = spsolve(
        build_laplacian_dirichlet(n_steps)
        + diags(fun_c(grid_interior)),
        rhs)
    grid = np.linspace(0, 1, n_steps + 1)
    
    if true_sol is None:
        res = (sol, grid)
    else:
        true_val = true_sol(grid)
        res = norm(sol - true_val, np.inf)
        if plot:
            plt.plot(grid, sol, 'o', color='b', label='approx')
            plt.plot(grid, true_val, color='r', label='truth')
            plt.legend()
            plt.show()
    
    return res

def neumann_v1(params, n_steps, true_sol=None, plot=False):
    fun_f, l_val, sigma = params
    step = 1 / n_steps
    grid_left = np.linspace(step, 1, n_steps) 
    
    rhs = fun_f(grid_left)
    rhs[0] += l_val * step ** -2
    rhs[-1] = sigma
    
    sol = np.zeros(n_steps + 1)
    sol[0] = l_val
    sol[1:] = spsolve(build_laplacian_neumann(n_steps), rhs)
    grid = np.linspace(0, 1, n_steps + 1)
    
    if true_sol is None:
        res = (sol, grid)
    else:
        true_val = true_sol(grid)
        res = norm(sol - true_val, np.inf)
        if plot:
            plt.plot(grid, sol, 'o', color='b', label='approx')
            plt.plot(grid, true_val, color='r', label='truth')
            plt.legend()
            plt.show()
    
    return res

def neumann_v2(params, n_steps, true_sol=None, plot=False):
    fun_f, l_val, sigma = params
    step = 1 / n_steps
    grid_left = np.linspace(step, 1, n_steps)

    rhs = fun_f(grid_left)
    rhs[0] += l_val * step ** -2
    rhs[-1] = sigma + 0.5 * step * fun_f(1)
    
    sol = np.zeros(n_steps + 1)
    sol[0] = l_val  
    sol[1:] = spsolve(build_laplacian_neumann(n_steps), rhs)
    grid = np.linspace(0, 1, n_steps + 1)
    
    if true_sol is None:
        res = (sol, grid)
    else:
        true_val = true_sol(grid)
        res = norm(sol - true_val, np.inf)
        if plot:
            plt.plot(grid, sol, 'o', color='b', label='approx')
            plt.plot(grid, true_val, color='r', label='truth')
            plt.legend()
            plt.show()
    
    return res

params_dir = (
    lambda val: val,
    lambda val: (1 + 2 * val - val ** 2) * np.exp(val),
    1,
    0)
sol_dir = (lambda val: (1 - val) * np.exp(val))
dirichlet(params_dir, 3, sol_dir, True)

params_neu = (lambda val: val, 1, -1)
sol_neu = (lambda val: (-1 / 6 * val ** 3 - 0.5 * val + 1))
neumann_v1(params_neu, 10, sol_neu, True)
neumann_v2(params_neu, 5, sol_neu, True)

n_steps_grid = 2 ** np.arange(3, 14)

errs = dict()
errs['Dirichlet'] = [
    dirichlet(params_dir, n_steps, sol_dir)
    for n_steps in n_steps_grid]
errs['Neumann v1'] = [
    neumann_v1(params_neu, n_steps, sol_neu)
    for n_steps in n_steps_grid]
errs['Neumann v2'] = [
    neumann_v2(params_neu, n_steps, sol_neu)
    for n_steps in n_steps_grid]

plot_orders(n_steps_grid, errs)
