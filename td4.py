#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.linalg import solve, norm
import matplotlib.pyplot as plt
from display_functions import (
    plot_orders, plot_graphs_2d, plot_energy)


## Newton's method

def newton(fun, grad_fun, val, ite_max=10, err_max=1e-10):
    err = err_max + 1
    ite = 0
    while err > err_max and ite < ite_max:
        img = fun(val)
        val = val - solve(grad_fun(val), img)
        err = norm(img, np.inf)
        ite += 1
    return val, ite

para = (lambda vec: [
    vec[0] ** 2 + vec[1] ** 2 - 2,
    vec[0] ** 2 - vec[1] ** 2 - 1])
grad_para = (lambda vec: [[
    2 * vec[0], 2 * vec[1]], [2 * vec[0], -2 * vec[1]]])

vals_newton, ites = zip(*[
    newton(para, grad_para, np.array([5, 4]), ite)
    for ite in range(0, 7)])
logs_errs = np.log([
    norm(val - np.array([1.5 ** 0.5, 2 ** -0.5]), 2)
    for val in vals_newton])
r_logs_errs = logs_errs[1:] / logs_errs[:-1]

plt.figure()
plt.plot(
    ites[:-1], r_logs_errs, marker='*',
    label=f"Newton's method, order {round(r_logs_errs[-1], 3)}")
plt.xlabel('number of iterations')
plt.ylabel('ratio log|x_{k + 1} - x| / log|x_k - x|')
plt.title("convergence of Newton's method")
plt.grid(True)
plt.legend()
plt.show()


## Numerical methods

# Schemes implementations

MET_REF = 'RK4'
implemented_methods = [
    'explicit Euler',
    'implicit Euler',
    'Crank Nicolson',
    'midpoint',
    'RK4']

class NumericalMethod():

    def __init__(self, name, n_steps, params, grad_fun=None):
        assert name in implemented_methods, 'not implemented'
        self.name = name
        self.n_steps = n_steps
        self.fun, self.val_init, self.end_time = params
        self.grad_fun = (
            None if method in ['explicit Euler', 'RK4']
            else grad_fun)
        self.step = self.end_time / n_steps

    def compute_gradient(self, val):
        mst = 0.5 * self.step
        fun = self.fun
        grad = self.grad_fun
        ide = np.eye(len(self.val_init))
        if self.name == 'implicit Euler':
            fun_ = (
                lambda var: -var + self.step * fun(var) + val)
            gra_ = lambda var: -ide + self.step * grad(var)
        elif self.name == 'Crank Nicolson':
            fun_ = (
                lambda var: -var
                + mst * (fun(var) + fun(val)) + val)
            gra_ = lambda var: -ide + mst * grad(var)
        elif self.name == 'midpoint':
            fun_ = (
                lambda var: -var +
                2 * mst * fun(.5 * (var + val)) + val)
            gra_ = (
                lambda var: -ide + mst * grad(.5 * (var + val)))
        return fun_, gra_
    
    def next_val(self, val):
        fun, step = self.fun, self.step
        if self.name == 'explicit Euler':
            val_new = val + step * fun(val)
        elif self.name == 'RK4':
            pt1 = fun(val)
            pt2 = fun(val + 0.5 * step * pt1)
            pt3 = fun(val + 0.5 * step * pt2)
            pt4 = fun(val + step * pt3)
            val_new = val + step / 6 * (
                pt1 + 2 * pt2 + 2 * pt3 + pt4)
        else:
            loc_fun, loc_grad = self.compute_gradient(val)
            val_new = newton(loc_fun, loc_grad, val)[0]
        return val_new

    def apply_scheme(self):
        vals = np.zeros((self.n_steps + 1, len(self.val_init)))
        vals[0] = self.val_init
        for n_step in range(0, self.n_steps):
            vals[n_step + 1] = self.next_val(vals[n_step])
        times = np.linspace(0, self.end_time, self.n_steps + 1)
        return vals, times, vals[self.n_steps]

# Example of the Lotka-Volterra system

PAR_A = 3
PAR_B = 1
PAR_C = 2
PAR_D = 1

fun_lv = (lambda vec: np.array([
    PAR_A * vec[0] - PAR_B * vec[0] * vec[1],
    -PAR_C * vec[1] + PAR_D * vec[0] * vec[1]]))

grad_lv = (lambda vec: np.array([
    [PAR_A - PAR_B * vec[1], -PAR_B * vec[0]],
    [PAR_D * vec[1], -PAR_C + PAR_D * vec[0]]]))

inte = (
    lambda grid:
    PAR_D * grid[:, 0] - PAR_C * np.log(grid[:, 0])
    + PAR_B * grid[:, 1] - PAR_A * np.log(grid[:, 1]))

vals_lv = dict()
x_vals_lv = dict()
y_vals_lv = dict()
intes_lv = dict()

params_lv = (fun_lv, np.array([1, 2]), 8)

for method in implemented_methods:
    scheme = NumericalMethod(method, 100, params_lv, grad_lv)
    vals_lv_loc, times_lv, _ = scheme.apply_scheme()
    vals_lv[method] = vals_lv_loc
    x_vals_lv[method] = vals_lv_loc[:, 0]
    y_vals_lv[method] = vals_lv_loc[:, 1]
    intes_lv[method] = inte(vals_lv_loc)

x_lim = min(x_vals_lv[MET_REF]) + max(x_vals_lv[MET_REF])
y_lim = min(y_vals_lv[MET_REF]) + max(y_vals_lv[MET_REF])

x_mesh, y_mesh = np.meshgrid(
    np.linspace(0, x_lim, 10), np.linspace(0, y_lim, 10))
fx_mesh, fy_mesh = fun_lv([x_mesh, y_mesh])

plt.figure(figsize=(15, 10))
plt.quiver(x_mesh, y_mesh, fx_mesh, fy_mesh, headlength=7)
for method in implemented_methods:
    plt.plot(x_vals_lv[method], y_vals_lv[method], label=method)
plt.xlim(0, x_lim)
plt.ylim(0, y_lim)
plt.grid(True)
plt.xlabel('number of preys')
plt.ylabel('number of predators')
plt.title('phase diagram')
plt.legend()
plt.show()

plot_graphs_2d(vals_lv, times_lv, ['preys', 'predators'])

plot_energy(intes_lv, times_lv)


# Orders of convergence of the schemes

n_steps_grid = 2 ** np.arange(9, 13)

ref_last_val = (
    NumericalMethod(MET_REF, 10 * 2 ** 13, params_lv, grad_lv)
    .apply_scheme()[2])

errs = dict()
for method in implemented_methods:
    errs[method] = [
        norm(
            NumericalMethod(method, n_steps, params_lv, grad_lv)
            .apply_scheme()[2]
            - ref_last_val, np.inf)
        for n_steps in n_steps_grid]
    
plot_orders(n_steps_grid, errs)


# Energy evolution for the harmonic oscillator

fun_ho = (lambda vec: np.array([vec[1], -vec[0]]))
grad_ho = (lambda vec: np.array([[0, 1], [-1, 0]]))
inte = (lambda grid: grid[:, 0] ** 2 + grid[:, 1] ** 2)
params_ho = (fun_ho, np.array([1, 2]), 8)

intes_ho = dict()
for method in implemented_methods:
    scheme = NumericalMethod(method, 100, params_ho, grad_ho)
    vals_ho_loc, times_ho, _ = scheme.apply_scheme()
    intes_ho[method] = inte(vals_ho_loc)
plot_energy(intes_ho, times_ho)
