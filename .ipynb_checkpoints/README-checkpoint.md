# pylmd
Localized Marker Detection (LMD) Python Implementation with GPU Acceleration

# THIS PROJECT IS A WIP

The inspiration for this project can be found at: https://www.nature.com/articles/s42003-025-08485-y.

After cloning repo, use pip install -e . to install package.
After cloning repo, use pip install -e .[gpu] to install package with GPU accelerated libraries.

Please ensure that CUDA is functioning (run nvidia-smi & nvcc) prior to installation of GPU version of this package.

Uses GPU for acceleration (if available). This requires an existing CUDA installation.

To use in jupyter:
from pylmd import lmd
