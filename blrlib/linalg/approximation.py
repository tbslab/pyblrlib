# -*- coding: utf-8 -*-
import numpy
from .. import core


def truncated_svd(mat, eps=None, rank=None):
    """Return two matrices approximated by using truncated singular
    value decomposition.

    Parameters
    ----------
    mat: array_like
        A matrix object which is of array_like.
    eps: float, default None
        Numerical value for controlled accuracy.
    rank: int, default None
        Numerical rank for fixed rank approximation.

    Returns
    -------
    U: matrix
        A left matrix of lrmatrix.
    V: matrix
        A right matrix of lrmatrix.
    """
    if rank:
        return _truncated_svd_fixed_rank(mat, rank)
    if eps:
        return _truncated_svd_controlled_accuracy(mat, eps)
    else:
        raise ValueError("'eps' or 'rank' parameter must be given")


def _truncated_svd_controlled_accuracy(mat, eps):
    """Return two matrices approximated by using truncated singular
    value decomposition.

    This is a approximation for controlled accuracy.
    """
    U, s, Vh = numpy.linalg.svd(mat)
    accuracy_bound = eps * numpy.linalg.norm(s)
    rank = 1

    while numpy.linalg.norm(s[rank:]) >= accuracy_bound:
        rank += 1

    return core.matrix(U[:, :rank] * s[:rank]), core.matrix(Vh[:rank, :])


def _truncated_svd_fixed_rank(mat, rank):
    """Return two matrices approximated by using truncated singular
    value decomposition.

    This is a approximation for fixed rank.
    """
    U, s, Vh = numpy.linalg.svd(mat)
    return core.matrix(U[:, :rank] * s[:rank]), core.matrix(Vh[:rank, :])


def aca(mat, eps=None, rank=None):
    """Return two matrices approximated by using adaptive cross
    approximation.

    Parameters
    ----------
    mat: array_like
        A matrix object which is of array_like.
    eps: float, default None
        Numerical value for controlled accuracy.
    rank: int, default None
        Numerical rank for fixed rank approximation.

    Returns
    -------
    U: matrix
    V: matrix
        A right matrix of lrmatrix.
    """
    if rank:
        return _aca_fixed_rank(mat, rank)
    if eps:
        return _aca_controlled_accuracy(mat, eps)
    else:
        raise ValueError("'eps' or 'rank' parameter must be given")


def _aca_controlled_accuracy(mat, eps):
    """Return two matrices approximated by using adaptive cross
    approximation.

    This is a approximation for controlled accuracy.
    """
    A = numpy.array(mat)
    max_rank = min(A.shape)
    pivot = numpy.random.randint(0, A.shape[1])
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

    return core.matrix(U), core.matrix(V)


def _aca_fixed_rank(mat, rank):
    """Return two matrices approximated by using adaptive cross
    approximation.

    This is a approximation for fixed rank.
    """
    A = numpy.array(mat)
    pivot = numpy.random.randint(0, A.shape[1])
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

    return core.matrix(U), core.matrix(V)


def _argmax_from_exclusion(a, indices):
    """Return argmax of 1D array excluding the indices you specified.

    Parameters
    ----------
    a: array_like
        1 dimensional array_like object.
    indices: set
        The index set you want to exclude.

    Returns
    -------
    index: int
        Argmax from exclusion.
    """
    for index in numpy.array(a).argsort()[::-1]:
        if index not in indices:
            return index
