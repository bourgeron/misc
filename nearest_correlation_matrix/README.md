The package `nearcorrmat` provides fast computations of the nearest correlation matrix, in the Euclidian sense.

# Package nearcorrmat

A correlation matrix is a real symmetric positive semidefinite matrix with unit diagonal.
The problem is to find the closest correlation matrix to a given positive semidefinite matrix, for the Frobenius norm. Five algorithms are implemented in the main function *nearest_corr*. Although the algorithms were not published recently, I am not aware of any freely accessible implementation, apart from the Python package `Statsmodels` providing the first algorithm.

# Algorithms

The first two algorithms are based on the primal problem, while the last three are directly based on the dual problem.

1. The algorithm 'grad' is based on [Higham, 2002](https://www.maths.manchester.ac.uk/~higham/narep/narep369.pdf).
This algorithm is a modified Dykstra's projection algorithm.
2. The algorithms 'admm_v0' and 'admm_v1' are ADMMs. Two versions are implemented, depending on the order in which the projections are performed. The two algorithms are different but have similar performances. To the best of my knowledge, this is an original application of the ADMM.
3. The algorithm 'bfgs' is the BFGS algorithm applied to the dual problem, as developped in [Malick, 2004](https://hal.inria.fr/inria-00072409v2/document). The implementations are based on [BGLS textbook](https://link.springer.com/book/10.1007/978-3-540-35447-5).
4. The algorithm 'l_bfgs' is the limited BFGS algorithm applied to the dual problem, as suggested in Malick. The implementations are based on the same book.
5. The algorithm 'newton' is a Newton's method for the dual problem, developped in [Qi, Sun, 2006](http://www.personal.soton.ac.uk/hdqi/REPORTS/simax_06.pdf).

These algorihms were tested on random 100 x 100 matrices, for err_max = 1e-6 (*cf.* performances.ipynb), which should be enough for a large range of use cases. The ranking of the algorithms' speeds may be significantly different for other ranges of parameters, *cf.* table in Qi, Sun. On a given machine, the algorithms have the following performances.

| algo     | time  | n_iterations |
|----------|-------|--------------|
| grad     | 20 s  | 1500         |
| admm_v0  | 8 s   | 500          |
| admm_v1  | 8 s   | 500          |
| bfgs     | 1 s   | 100          |
| l-bfgs   | 2 s   | 100          |
| newton   | 60 s  | 100          |

# License

This project is under MIT license.
