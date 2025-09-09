# pylmd
Localized Marker Detection (LMD) Python Implementation with GPU Acceleration

After cloning repo, use pip install -e pylmd to install package.

Uses GPU for acceleration (if available). This requires an existing CUDA installation.

To use in jupyter:
from pylmd.lmd import lmd

NOTE: Please use scanpy / rapids-singlecell for data pre-processing (normalization, scaling, PCA, etc).