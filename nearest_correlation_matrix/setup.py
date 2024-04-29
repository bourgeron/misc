# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['nearcorrmat']

package_data = \
{'': ['*']}

install_requires = \
['numpy>=1.21.2,<2.0.0']

setup_kwargs = {
    'name': 'nearcorrmat',
    'version': '0.1.0',
    'description': 'Find the nearest correlation matrix to a given one',
    'long_description': "The package `nearcorrmat` provides fast computations of the nearest correlation matrix, in the Euclidian sense.\n\n# Package nearcorrmat\n\nA correlation matrix is a real symmetric positive semidefinite matrix with unit diagonal.\nThe problem is to find the closest correlation matrix to a given positive semidefinite matrix, for the Frobenius norm. Five algorithms are implemented in the main function *nearest_corr*. Although the algorithms were not published recently, I am not aware of any freely accessible implementation, apart from the Python package `Statsmodels` providing the first algorithm.\n\n# Easy install and Source code\n\nTo install the package, it is enough to run: `pip install nearcorrmat`\nThe source code is available on `https://github.com/bourgeron/numerical_analysis/tree/master/nearest_corr_mat`\n\n# Algorithms\n\nThe first two algorithms are based on the primal problem, while the last three are directly based on the dual problem.\n\n1. The algorithm 'grad' is based on [Higham, 2002](https://www.maths.manchester.ac.uk/~higham/narep/narep369.pdf).\nThis algorithm is a modified Dykstra's projection algorithm.\n2. The algorithms 'admm_v0' and 'admm_v1' are ADMMs. Two versions are implemented, depending on the order in which the projections are performed. The two algorithms are different but have similar performances. To the best of my knowledge, this is an original application of the ADMM.\n3. The algorithm 'bfgs' is the BFGS algorithm applied to the dual problem, as developped in [Malick, 2004](https://hal.inria.fr/inria-00072409v2/document). The implementations are based on [BGLS textbook](https://link.springer.com/book/10.1007/978-3-540-35447-5).\n4. The algorithm 'l_bfgs' is the limited BFGS algorithm applied to the dual problem, as suggested in Malick. The implementations are based on the same book.\n5. The algorithm 'newton' is a Newton's method for the dual problem, developped in [Qi, Sun, 2006](http://www.personal.soton.ac.uk/hdqi/REPORTS/simax_06.pdf).\n\nThese algorihms were tested on random 100 x 100 matrices, for err_max = 1e-6 (*cf.* performances.ipynb), which should be enough for a large range of use cases. The ranking of the algorithms' speeds may be significantly different for other ranges of parameters, *cf.* table in Qi, Sun. On a given machine, the algorithms have the following performances.\n\n| algo     | time  | n_iterations |\n|----------|-------|--------------|\n| grad     | 20 s  | 1500         |\n| admm_v0  | 8 s   | 500          |\n| admm_v1  | 8 s   | 500          |\n| bfgs     | 1 s   | 100          |\n| l-bfgs   | 2 s   | 100          |\n| newton   | 60 s  | 100          |\n\n# License\n\nThis project is under MIT license.\n",
    'author': 'Thibault Bourgeron',
    'author_email': 'thibault.bourgeron@protonmail.com',
    'maintainer': 'Thibault Bourgeron',
    'maintainer_email': 'thibault.bourgeron@protonmail.com',
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7,<3.11',
}


setup(**setup_kwargs)
