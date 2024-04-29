#!/usr/bin/env python
# coding: utf-8

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from display_functions import plot_orders

def build_laplacian_2d(n_steps):
    step = 1 / n_steps
    dim = (n_steps - 1) ** 2
    diag0 = 4 * np.ones(dim)
    diag1 = -np.ones(dim - 1)
    diag1[n_steps - 2:dim:n_steps - 1] = 0
    diagn = -np.ones(dim - n_steps + 1)
    pentadiag = step ** -2 * diags(
        [diagn, diag1, diag0, diag1, diagn],
        [-n_steps + 1, -1, 0, 1, n_steps - 1], format='lil')
    return pentadiag.tocsr()

def dirichlet_2d(fun_f, n_steps, true_sol=None, plot=False):
    step = 1 / n_steps
    grid_interior = np.linspace(step, 1 - step, n_steps - 1)
    x_mesh, y_mesh = np.meshgrid(grid_interior, grid_interior)

    sol = np.zeros((n_steps + 1, n_steps + 1))
    sol[1:n_steps, 1:n_steps] = (
        spsolve(
            build_laplacian_2d(n_steps),
            fun_f(x_mesh.reshape(-1, 1), y_mesh.reshape(-1, 1)))
        .reshape(n_steps - 1, n_steps - 1))

    grid = np.linspace(0, 1, n_steps + 1)
    x_mesh, y_mesh = np.meshgrid(grid, grid)

    if true_sol is None:
        res = (sol, x_mesh, y_mesh)
    else:
        true_val = true_sol(x_mesh, y_mesh)
        res = norm(sol - true_val, np.inf)   
        if plot:
            plt.contour(
                x_mesh, y_mesh, sol, n_steps + 1, colors='k')
            plt.contourf(x_mesh, y_mesh, sol, n_steps + 1)
            plt.axis('scaled')
            plt.title(
                'Solution on a grid {0} x {0}'
                .format(n_steps))
            plt.show()
    return res

fun_bnd = (
    lambda val_x, val_y:
    np.sin(np.pi * val_x) * np.sin(np.pi * val_y))
sol_dir = (
    lambda val_x, val_y:
    0.5 * np.pi ** -2 * fun_bnd(val_x, val_y))
dirichlet_2d(fun_bnd, 15, sol_dir, True)

n_steps_grid = 2 ** np.arange(2, 7)
errs = {
    'Dirichlet 2D': [
        dirichlet_2d(fun_bnd, n_steps, sol_dir)
        for n_steps in n_steps_grid]}
plot_orders(n_steps_grid, errs)
