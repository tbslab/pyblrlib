# -*- coding: utf-8 -*-
"""
    blrlib
    ======
    The Package for Block Low Rank Matrix Computations.
"""

from . import core
from . import io
from . import linalg
from . import vis

from .core.vector import Vector
from .core.matrix import Zero
from .core.matrix import Dense
from .core.matrix import LowRank
from .core.matrix import BlockLowRank
from .core.matrix import build

__all__ = []
__all__.extend(core.__all__)
__all__.extend(io.__all__)
__all__.extend(linalg.__all__)
__all__.extend(vis.__all__)

__version__ = "0.2.0"
