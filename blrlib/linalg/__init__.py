# -*- coding: utf-8 -*-
"""
    blrlib.linalg
    =============
    The Subpackage for blrmat package.
"""
from .factorization import qr
from .approximation import truncated_svd, aca

__all__ = ["qr", "truncated_svd", "aca"]
