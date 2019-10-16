# -*- coding: utf-8 -*-
"""
    blrlib.core
    ===========
    The Subpackage for Block Low Rank Matrix Computations.
"""
from .vec import vector
from .mat import zmatrix
from .mat import matrix
from .mat import lrmatrix
from .mat import blrmatrix
from .mat import build_blrmatrix

__all__ = ["vector",
           "zmatrix", "matrix", "lrmatrix", "blrmatrix",
           "build_blrmatrix"]
