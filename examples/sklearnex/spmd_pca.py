import numpy as np
import scipy as sp
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))

sys.path.append(parent)

import dpctl
from sklearn.utils._testing import assert_allclose
from onedal.spmd.decomposition import PCA as PCA_Spmd


def test_sklearnex_import():
    from sklearnex.preview.decomposition import PCA
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA_Spmd(n_components=2, svd_solver='full').fit(X)
    assert 'sklearnex' in pca.__module__
    assert hasattr(pca, '_onedal_estimator')
    assert_allclose(pca.singular_values_, [6.30061232, 0.54980396])

if __name__ == "__main__":
    q = dpctl.SyclQueue("gpu")