# pylmd (Python)

GPU-accelerated reimplementation of Localized Marker Detector (LMD) for Python.

- Input: AnnData object
- Acceleration: tries RAPIDS if GPU available; falls back to NumPy/SciPy/Scikit-learn

Install

```
# CPU-only
pip install -e pylmd

# With GPU acceleration (RAPIDS)
pip install -e pylmd[gpu]
```

Python Usage

```python
from pylmd.lmd import lmd

res = lmd(adata, n_neighbors=30, use_rep='X_pca', correction=False)
# res contains 'score_profile', 'lmds', 'rank', 'knee_index'
```
