# -*- coding: utf-8 -*-
import numpy
from .. import core


def qr(obj):
    """Return QR factorization for BlockLowRank or Dense object.

    Parameters
    ----------
    obj: BlockLowRank or Dense
        A matrix to be factorized.

    Returns
    -------
    Q: BlockLowRank or Dense
        A Dense with orthonormal columns.
    R: BlockLowRank or Dense
        The upper triangular Dense.
    """
    if isinstance(obj, (core.Dense, numpy.ndarray)):
        Q, R = numpy.linalg.qr(obj)
        return core.Dense(Q), core.Dense(R)
    if isinstance(obj, core.BlockLowRank):
        return _blr_mbgs(obj)
    return NotImplemented


def _blr_tsqr(obj):
    """Return Tall-Skinny QR factorization for BLR matrices.

    Parameters
    ----------
    obj: BlockLowRank
        A matrix to be factorized. The block shape must be (nb, 1).

    Returns
    -------
    Q: BlockLowRank
        A BLR matrix with orthonormal columns.
        objects.
    R: Dense
        The upper triangular matrix.
    """
    nb = obj.nb[0]
    A = obj
    Q = core.BlockLowRank(numpy.full((nb, 1), None))
    B = numpy.full(nb, None)

    for i in range(nb):
        if isinstance(A[i, 0], core.LowRank):
            Qi, Ri = qr(A[i, 0].U)
            Q[i, 0] = Qi
            B[i] = Ri * A[i, 0].V
        else:
            B[i] = A[i, 0]

    B = numpy.vstack(B)

    if B.shape[0] < B.shape[1]:
        Z = numpy.zeros((B.shape[1] - B.shape[0], B.shape[1]))
        B = numpy.vstack([B, Z])

    Qb, R = qr(B)
    rstart, rend = 0, 0

    for i in range(nb):
        if isinstance(A[i, 0], core.LowRank):
            rstart = rend
            rend = rend + A[i, 0].rank
            U = Q[i, 0]
            V = Qb[rstart:rend, :]
            Q[i, 0] = core.LowRank((U, V), A[i, 0].method, A[i, 0].eps)
        else:
            rstart = rend
            rend = rend + A[i, 0].shape[0]
            Q[i, 0] = Qb[rstart:rend, :]

    return Q, R


def _blr_mbgs(obj):
    """Return Modified block Gram-Schmidt algorithm for BLR matrices.

    Parameters
    ----------
    obj: BlockLowRank
        A matrix to be factorized.

    Returns
    -------
    Q: BlockLowRank
        A BLR matrix with orthonormal columns.
    R: BlockLowRank
        The upper triangular BLR matrix.
    """
    rnb, cnb = obj.nb
    min_nb = min(obj.nb)
    A = obj.copy()
    Q = core.BlockLowRank(numpy.full((rnb, min_nb), None))
    R = core.BlockLowRank(numpy.full((min_nb, cnb), None))

    for i, j in numpy.ndindex(R.nb):
        rows = A[i, i].shape[1]
        cols = A[i, j].shape[1]
        R[i, j] = core.Zero((rows, cols))

    for j in range(min_nb):
        Q[:, j], R[j, j] = _blr_tsqr(A[:, j])

        for k in range(j + 1, cnb):
            R[j, k] = (Q[:, j].T * A[:, k])[0, 0]
            A[:, k] = A[:, k] - Q[:, j] * core.BlockLowRank([[R[j, k]]])

    return Q, R
