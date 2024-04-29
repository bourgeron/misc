#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def plot_orders(n_steps_grid, errs):
    log_errs = {name: np.log(err) for name, err in errs.items()}
    log_n_steps_grid = np.log(n_steps_grid)
    
    plt.figure(figsize=(15, 10))
    for name, color in zip(log_errs, list('bgrcm')):
        log_errs_loc = log_errs[name]
        reg = np.polyfit(log_n_steps_grid, log_errs_loc, 1)
        plt.plot(
            log_n_steps_grid, log_errs_loc, 'o', color=color)
        plt.plot(
            log_n_steps_grid,
            reg[0] * log_n_steps_grid + reg[1],
            color=color,
            label=f'{name}, order {round(-reg[0], 3)}')
    plt.ylabel('error')
    plt.xlabel('grid size')
    plt.grid(True)
    plt.title('log-log scale')
    plt.legend()
    plt.show()

def plot_graphs_2d(vals, times, labels=None, colors=None):
    labels = ['x', 'y'] if labels is None else labels
    colors = ['b', 'g'] if colors is None else colors
    for name, values in vals.items():
        plt.figure()
        for val in zip(*values, colors, labels):
            plt.plot(
                times,
                val[:-2],
                color=val[-2],
                label=val[-1])
        plt.title(f'{name}')
        plt.xlabel('time')
        plt.ylabel('values')
        plt.legend()
        plt.show()

def plot_energy(intes, times):
    plt.figure()
    for name in intes:
        plt.plot(times, intes[name], label=name)
    plt.title('evolution of the energy')
    plt.xlabel('time')
    plt.ylabel('values of the energy')
    plt.legend()
    plt.show()

    plt.figure()
    for name, color in [['Crank Nicolson', 'r'], ['midpoint', 'c']]:
        plt.plot(times, intes[name], label=name, color=color)
    plt.title('evolution of the energy')
    plt.xlabel('time')
    plt.ylabel('values of the energy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(
        times,
        intes['RK4'],
        label='RK4',
        color='m')
    plt.title('evolution of the energy')
    plt.xlabel('time')
    plt.ylabel('values of the energy')
    plt.legend()
    plt.show()

def save_anim(scheme, name, n_steps_space, params):
    grid_space, vals, val_init, alpha, times = scheme(
        name, n_steps_space, params)

    def update(time):
        lines[0].set_data(
            grid_space, np.roll(val_init, int(time * alpha)))
        lines[1].set_data(grid_space, vals[time])
        axis.set_title(
            f'{name} scheme, time {round(times[time], 2)}')
        return lines

    fig, axis = plt.subplots()
    axis.set_xlim((0, 1))
    axis.set_ylim((min(val_init) - 0.2, max(val_init) + 0.2))
    lines = [
        axis.plot([], label='exact solution')[0],
        axis.plot([], label='approximated solution')[0]]
    axis.legend()
    axis.grid()

    anim = animation.FuncAnimation(
        fig, update, frames=len(vals),
        interval=20, blit=True, repeat=False)
    writer = (animation.writers['ffmpeg'])(fps=30, bitrate=1800)
    anim.save(f'{name}.mp4', writer=writer)
