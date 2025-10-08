from __future__ import annotations

import numpy as np

def np_as_dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)

