# -*- coding: utf-8 -*-
import numpy
from .. import core


def svda(mat, eps=None, rank=None):
    """Return two matrices approximated by using singular value decomposition (SVD).

    Arguments
    ---------
    mat : array like
        A matrix object which is of array like.
    eps : float, optional
        Numerical value for controlled accuracy.
    rank : int, optional
        Numerical rank for fixed rank approximation.

    Returns
    -------
    left : matrix
        A left matrix of lrmatrix.
    right : matrix
        A right matrix of lrmatrix.
    """
    if rank:
        return _svda_fixed_rank(mat, rank)
    if eps:
        return _svda_controlled_accuracy(mat, eps)


def _svda_controlled_accuracy(mat, eps):
    """Return two matrices approximated by using singular value decomposition (SVD).

        This is a approximation for controlled accuracy.
    """
    U, s, Vh = numpy.linalg.svd(mat)
    accuracy_bound = eps * numpy.linalg.norm(s)
    rank = 1

    while numpy.linalg.norm(s[rank:]) >= accuracy_bound:
        rank += 1

    return core.matrix(U[:, :rank] * s[:rank]), core.matrix(Vh[:rank, :])


def _svda_fixed_rank(mat, rank):
    """Return two matrices approximated by using singular value decomposition (SVD).

        This is a approximation for fixed rank.
    """
    U, s, Vh = numpy.linalg.svd(mat)
    return core.matrix(U[:, :rank] * s[:rank]), core.matrix(Vh[:rank, :])


def aca(mat, eps=None, rank=None):
    """Return two matrices approximated by using adaptive cross approximation (ACA).

    Arguments
    ---------
    mat : array like
        A matrix object which is of array like.
    eps : float, optional
        Numerical value for controlled accuracy.
    rank : int, optional
        Numerical rank for fixed rank approximation.

    Returns
    -------
    left : matrix
        A left matrix of lrmatrix.
    right : matrix
        A right matrix of lrmatrix.
    """
    if rank:
        return _aca_fixed_rank(mat, rank)
    if eps:
        return _aca_controlled_accuracy(mat, eps)


def _aca_controlled_accuracy(mat, eps):
    """Return two matrices approximated by using adaptive cross approximation (ACA).

        This is a approximation for controlled accuracy.
    """
    pass


def _aca_fixed_rank(mat, rank):
    """Return two matrices approximated by using adaptive cross approximation (ACA).

        This is a approximation for fixed rank.
    """
    pass
