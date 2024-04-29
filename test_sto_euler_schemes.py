from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from sto_euler_schemes import (
    gen_rand_cov_mat,
    homogeneous,
    path_independent,
    time_independent,
    generic,
    generate_path_euler
)


def mean_gbm(time):
    return 1e-3 * np.sin(time)

def vol_gbm(time):
    return 0.3 + 0.1 * np.cos(time)

def gbm_generic_form(dtime, time, dbrown, x_t, *, mean, vol):
    return (mean(time) * dtime + vol(time) * dbrown) * x_t

def bachelier_generic_form(dtime, time, dbrown, x_t, *, mean, vol):
    del x_t
    return gbm_generic_form(dtime, time, dbrown, 1, mean=mean, vol=vol)

bachelier = partial(bachelier_generic_form, x_t=None)

def cir(dtime, time, dbrown, r_t, strenght, long_term_mean, vol):
    del time
    return (
        strenght * (long_term_mean - r_t) * dtime
        + vol * r_t ** 0.5 * dbrown)  # time independent

def hull_white(dtime, time, dbrown, r_t, theta, alpha, vol):
    return (theta(time) - alpha(time) * r_t) * dtime + vol(time) * dbrown

schemes = {
    'gbm': (bachelier, homogeneous, 'Geometric Brownian Motion'),
    'gbm_generic_form': (gbm_generic_form, generic),
    'bachelier': (bachelier, path_independent, 'Bachelier Process'),
    'bachelier_generic_form': (bachelier_generic_form, generic),
    'cir': (cir, time_independent, 'Cox-Ingersoll-Ross Process'),
    'cir_generic_form': (cir, generic),
    'hull_white': (hull_white, generic, 'Hull-White Process'),
}

if __name__ == '__main__':
    from loguru import logger
    inits = [2, 3, 4, 5]
    cov = gen_rand_cov_mat(len(inits))
    logger.info('covariance matrix')
    logger.info(cov)
    logger.info('\n')

    def check_coherence(scheme_name, scheme_specs, init_vals, cov_mat, **kwargs):
        logger.info(scheme_name)
        logger.info(generate_path_euler(
            *scheme_specs[:2], init_vals, 10, 10, cov_mat, 0, **kwargs)[1])
        logger.info('\n')

    for name, specs in schemes.items():
        if 'b' in name:
            extras = {'vol': vol_gbm, 'mean': mean_gbm}
            check_coherence(name, specs, inits, cov, **extras)

    for name, specs in schemes.items():
        if 'cir' in name:
            extras = {'strenght': 1, 'long_term_mean': 2, 'vol': 0.1}
            check_coherence(name, specs, inits, cov, **extras)

    def plot_scheme_path(scheme_name, scheme_specs, init_vals, cov_mat, save=False, **kwargs):
        plt.figure(figsize=(16, 9))
        plt.plot(*generate_path_euler(
            *scheme_specs[:2], init_vals, 10, 1000, cov_mat, **kwargs))
        plt.title(scheme_specs[2])
        if save:
            plt.savefig(f'{scheme_name}.png', dpi=300)
        plt.grid()
        plt.show()

    plot_params = {'mean': mean_gbm, 'vol': vol_gbm}
    plot_scheme_path('gbm', schemes['gbm'], inits, cov, **plot_params)

    plot_params = {'mean': mean_gbm, 'vol': vol_gbm}
    plot_scheme_path('bachelier', schemes['bachelier'], inits, cov, **plot_params)

    plot_params = {'strenght': 1, 'long_term_mean': 2, 'vol': 0.1}
    plot_scheme_path('cir', schemes['cir'], inits, cov, **plot_params)

    plot_params = {'theta': mean_gbm, 'alpha': vol_gbm, 'vol': vol_gbm}
    plot_scheme_path('hull_white', schemes['hull_white'], inits, cov, **plot_params)
