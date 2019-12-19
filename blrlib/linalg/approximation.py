# -*- coding: utf-8 -*-
import numpy
from .. import core


def truncated_svd(obj, eps=None, rank=None):
    """Return two matrices approximated by using truncated singular
    value decomposition.

    Arguments:
        obj (array_like): 2 dimensinal array object.
        eps (float, optional): Numerical value for adaptive rank approximation.
        rank (int, optional): Numerical rank for fixed rank approximation.

    Returns:
        Dense: A left matrix of ``LowRank`` object.
        Dense: A right matrix of ``LowRank`` object.
    """
    if rank:
        return _truncated_svd_fixed(obj, rank)
    if eps:
        return _truncated_svd_adaptive(obj, eps)
    else:
        raise ValueError("'eps' or 'rank' argument must be given")


def _truncated_svd_adaptive(obj, eps):
    """Return two matrices approximated by using truncated singular
    value decomposition.
    """
    U, s, Vh = numpy.linalg.svd(obj)
    rank = 1

    while s[rank] >= eps:
        rank += 1

    return core.Dense(U[:, :rank] * s[:rank]), core.Dense(Vh[:rank, :])


def _truncated_svd_fixed(obj, rank):
    """Return two matrices approximated by using truncated singular
    value decomposition.
    """
    U, s, Vh = numpy.linalg.svd(obj)
    return core.Dense(U[:, :rank] * s[:rank]), core.Dense(Vh[:rank, :])


def aca(obj, eps=None, rank=None):
    """Return two matrices approximated by using adaptive cross
    approximation.

    Arguments:
        obj (array_like): 2 dimensinal array object.
        eps (float, optional): Numerical value for adaptive rank approximation.
        rank (int, optional): Numerical rank for fixed rank approximation.

    Returns:
        Dense: A left matrix of ``LowRank`` object.
        Dense: A right matrix of ``LowRank`` object.
    """
    if rank:
        return _aca_fixed(obj, rank)
    if eps:
        return _aca_adaptive(obj, eps)
    else:
        raise ValueError("'eps' or 'rank' argument must be given")


def _aca_adaptive(obj, eps):
    """Return two matrices approximated by using adaptive cross
    approximation.
    """
    A = numpy.array(obj)
    max_rank = min(A.shape)
    pivot = 0
    pivot_cols = {pivot}
    u = numpy.array(A[:, pivot], dtype=numpy.float)
    pivot = numpy.abs(u).argmax()
    pivot_rows = {pivot}
    u = u / u[pivot]
    v = numpy.array(A[pivot, :], dtype=numpy.float)
    U = u.reshape((u.shape[0], 1))
    V = v.reshape((1, v.shape[0]))
    pivot = _argmax_from_exclusion(numpy.abs(v), pivot_cols)
    pivot_cols.add(pivot)
    nu = numpy.linalg.norm(u) * numpy.linalg.norm(v)
    mu2 = nu ** 2
    r = 1

    while nu >= eps * numpy.sqrt(mu2) and r < max_rank:
        u = A[:, pivot] - U @ V[:, pivot]
        pivot = _argmax_from_exclusion(numpy.abs(u), pivot_rows)
        pivot_rows = {pivot}
        u = u / u[pivot]
        v = A[pivot, :] - U[pivot, :] @ V
        U = numpy.hstack([U, u.reshape((u.shape[0], 1))])
        V = numpy.vstack([V, v.reshape((1, v.shape[0]))])
        pivot = _argmax_from_exclusion(numpy.abs(v), pivot_cols)
        pivot_cols.add(pivot)
        nu = numpy.linalg.norm(u) * numpy.linalg.norm(v)
        tmp = 0
        for j in range(r):
            tmp += numpy.dot(U[:, j], u) * numpy.dot(V[j, :], v)
        mu2 = mu2 + nu ** 2 + 2 * tmp
        r = r + 1

    return core.Dense(U), core.Dense(V)


def _aca_fixed(obj, rank):
    """Return two matrices approximated by using adaptive cross
    approximation.
    """
    A = numpy.array(obj)
    pivot = 0
    pivot_cols = {pivot}
    u = numpy.array(A[:, pivot], dtype=numpy.float)
    pivot = numpy.abs(u).argmax()
    pivot_rows = {pivot}
    u = u / u[pivot]
    v = numpy.array(A[pivot, :], dtype=numpy.float)
    U = u.reshape((u.shape[0], 1))
    V = v.reshape((1, v.shape[0]))
    pivot = _argmax_from_exclusion(numpy.abs(v), pivot_cols)
    pivot_cols.add(pivot)
    r = 1

    while r < rank:
        u = A[:, pivot] - U @ V[:, pivot]
        pivot = _argmax_from_exclusion(numpy.abs(u), pivot_rows)
        pivot_rows = {pivot}
        u = u / u[pivot]
        v = A[pivot, :] - U[pivot, :] @ V
        U = numpy.hstack([U, u.reshape((u.shape[0], 1))])
        V = numpy.vstack([V, v.reshape((1, v.shape[0]))])
        pivot = _argmax_from_exclusion(numpy.abs(v), pivot_cols)
        pivot_cols.add(pivot)
        r = r + 1

    return core.Dense(U), core.Dense(V)


def _argmax_from_exclusion(a, indices):
    """Return argmax of 1D array excluding the indices you specified.

    Arguments:
        a (array_like): 1 dimensional array object.
        indices (set): The index set you want to exclude.

    Returns:
        int: Argmax from the exclusion.
    """
    for index in numpy.array(a).argsort()[::-1]:
        if index not in indices:
            return index
