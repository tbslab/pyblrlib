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

from .core.vec import vector
from .core.mat import zmatrix
from .core.mat import matrix
from .core.mat import lrmatrix
from .core.mat import blrmatrix
from .core.mat import build_blrmatrix

__all__ = []
__all__.extend(core.__all__)
__all__.extend(io.__all__)
__all__.extend(linalg.__all__)
__all__.extend(vis.__all__)

__version__ = "0.0.1"
