# -*- coding: utf-8 -*-
"""
    blrlib.linalg
    =============
    The Subpackage for blrmat package.
"""
from .factorization import qr
from .approximation import svda

__all__ = ["qr", "svda"]
