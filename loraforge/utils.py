"""Small utilities: seeding, threads."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_threads() -> None:
    n = os.cpu_count() or 2
    torch.set_num_threads(min(4, n))
    torch.set_num_interop_threads(1)
