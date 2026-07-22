"""
repro.py — make local randomness reproducible (V3 Stage 0: fixed-seed module).

The engines' own seeds are passed per call (see thesis_config.SEED). This module
pins randomness on OUR side, e.g. when sampling which queries enter the pilot.
Note: PYTHONHASHSEED only fully takes effect if set before interpreter start, so
we export it for child processes too.
"""

import os
import random


def set_global_seeds(seed: int) -> int:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    return seed
