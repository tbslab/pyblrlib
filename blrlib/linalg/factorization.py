# -*- coding: utf-8 -*-
import numpy
from .. import core


def qr(mat):
    """Return QR factorization for BlockLowRank or Dense object.

    Parameters
    ----------
    mat: BlockLowRank or Dense
        A matrix to be factorized.

    Returns
    -------
    Q: BlockLowRank or Dense
        A Dense with orthonormal columns.
    R: BlockLowRank or Dense
        The upper triangular Dense.
    """
    if isinstance(mat, (core.Dense, numpy.ndarray)):
        q, r = numpy.linalg.qr(mat)
        return core.Dense(q), core.Dense(r)
    if isinstance(mat, core.BlockLowRank):
        return _mbgs_for_blrmatrix(mat)
    return NotImplemented


def _tsqr_for_blrmatrix(blrmat):
    """Return Tall-Skinny QR factorization for BLR matrices.

    Parameters
    ----------
    blrmat: BlockLowRank
        A matrix to be factorized. The shape must be (rows, 1).

    Returns
    -------
    Q: numpy.ndarray
        A matrix with orthonormal columns. A list of Dense and LowRank
        objects.
    R: Dense
        The upper triangular matrix.
    """
    nb = blrmat.shape[0]
    X = blrmat
    Q = numpy.full(nb, None)
    B = numpy.full(nb, None)

    for i in range(nb):
        if isinstance(X.A[i, 0], core.LowRank):
            Qi, Ri = numpy.linalg.qr(X.A[i, 0].U)
            Q[i] = Qi
            B[i] = Ri @ X.A[i, 0].V
        else:
            B[i] = X.A[i, 0]

    B = numpy.vstack(B)

    if B.shape[0] < B.shape[1]:
        Z = numpy.zeros((B.shape[1] - B.shape[0], B.shape[1]))
        B = numpy.vstack([B, Z])

    Qb, R = numpy.linalg.qr(B)
    row1, row2 = 0, 0

    for i in range(nb):
        if isinstance(X.A[i, 0], core.LowRank):
            row1, row2 = row2, row2 + X.A[i, 0].rank
            Q[i] = core.LowRank((Q[i], Qb[row1:row2, :]))
        else:
            row1, row2 = row2, row2 + X.A[i, 0].shape[0]
            Q[i] = core.Dense(Qb[row1:row2, :])

    return Q, core.Dense(R)


def _mbgs_for_blrmatrix(blrmat):
    """Return Modified block Gram-Schmidt algorithm for BLR matrices.

    Parameters
    ----------
    blrmat: BlockLowRank
        A matrix to be factorized.

    Returns
    -------
    Q: BlockLowRank
        A BLR matrix with orthonormal columns.
    R: BlockLowRank
        The upper triangular BLR matrix.
    """
    nb = blrmat.shape[1]
    min_nb = min(blrmat.shape)
    X = core.BlockLowRank(blrmat.A.copy())
    Q = numpy.full((X.shape[0], min_nb), None)
    R = numpy.full((min_nb, X.shape[1]), None)

    for index in numpy.ndindex(R.shape):
        shape_row = X.A[index[0], index[0]].shape[1]
        shape_col = X.A[index].shape[1]
        R[index] = core.Zero((shape_row, shape_col))

    for j in range(nb):
        Q[:, j], R[j, j] = _tsqr_for_blrmatrix(X[:, j])

        for k in range(j + 1, nb):
            Qj = core.BlockLowRank(Q[:, j:j + 1])
            R[j, k] = (Qj.T @ X[:, k]).A[0, 0]
            X.A[:, k:k + 1] -= Qj.A @ R[j:j + 1, k:k + 1]

        if j >= min_nb - 1:
            return core.BlockLowRank(Q), core.BlockLowRank(R)

    return core.BlockLowRank(Q), core.BlockLowRank(R)
