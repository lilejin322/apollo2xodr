"""
Geometry module for the reverse engineering of apollo hdmap.
"""

import numpy as np

###################################### Gauss-Legendre ######################################
"""
The original Gauss-Legendre is approximate $\int_{-1}^{1} f(x)\,dx \approx \sum_{i=1}^{16} w_i f(x_i)$,
where $w_i$ are the weights and $x_i$ are the nodes. This module maps the original one to $[0, 1]$
"""
_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(16)  # Gauss-Legendre nodes and weights
_GAUSS_P = (_GAUSS_X + 1) / 2                             # mapped nodes on [0, 1]
_GAUSS_W = _GAUSS_W / 2                                   # mapped weights on [0, 1]
#############################################################################################
