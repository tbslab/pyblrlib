# -*- coding: utf-8 -*-
import numpy
from .. import core


def qr(mat):
    """Return QR factorization for blrmatrix or matrix.

    Parameters
    ----------
    mat : blrmatrix or matrix
        A matrix to be factored.

    Returns
    -------
    Q : blrmatrix or matrix
        A matrix with orthonormal columns.
    R : blrmatrix or matrix
        The upper triangular matrix.
    """
    if isinstance(mat, (core.matrix, numpy.ndarray)):
        q, r = numpy.linalg.qr(mat)
        return core.matrix(q), core.matrix(r)
    if isinstance(mat, core.blrmatrix):
        return _mbgs_for_blrmatrix(mat)
    return NotImplemented


def _tsqr_for_blrmatrix(blrmat):
    """Return Tall-Skinny QR factorization for BLR matrices.

    Parameters
    ----------
    blrmat : blrmatrix
        A matrix to be factored. The shape must be (nb, 1).

    Returns
    -------
    Q : numpy.ndarray
        A matrix with orthonormal columns. A list of matrix and lrmatrix object.
        objects.
    R : matrix
        The upper triangular matrix.
    """
    nb = blrmat.shape[0]
    X = blrmat
    Q = numpy.full(nb, None)
    B = numpy.full(nb, None)

    for i in range(nb):
        if isinstance(X.A[i, 0], core.lrmatrix):
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
    rs, re = 0, 0

    for i in range(nb):
        if isinstance(X.A[i, 0], core.lrmatrix):
            rs, re = re, re + X.A[i, 0].rank
            Q[i] = core.lrmatrix((Q[i], Qb[rs:re, :]))
        else:
            rs, re = re, re + X.A[i, 0].shape[0]
            Q[i] = core.matrix(Qb[rs:re, :])

    return Q, core.matrix(R)


def _mbgs_for_blrmatrix(blrmat):
    """Return Modified block Gram-Schmidt algorithm for BLR matrices.

    Parameters
    ----------
    blrmat : blrmatrix
        A matrix to be factored.

    Returns
    -------
    Q : blrmatrix
        A matrix with orthonormal columns.
    R : blrmatrix
        The upper triangular matrix.
    """
    nb = blrmat.shape[1]
    nb_min = min(blrmat.shape)
    X = core.blrmatrix(blrmat.A.copy())
    Q = numpy.full((X.shape[0], nb_min), None)
    R = numpy.full((nb_min, X.shape[1]), None)

    for index in numpy.ndindex(R.shape):
        rshape = X.A[index[0], index[0]].shape[1]
        cshape = X.A[index].shape[1]
        R[index] = core.zmatrix((rshape, cshape))

    for j in range(nb):
        Q[:, j], R[j, j] = _tsqr_for_blrmatrix(X[:, j])

        for k in range(j + 1, nb):
            Qj = core.blrmatrix(Q[:, j:j + 1])
            R[j, k] = (Qj.T @ X[:, k]).A[0, 0]
            X.A[:, k:k + 1] -= Qj.A @ R[j:j + 1, k:k + 1]

        if j >= nb_min - 1:
            return core.blrmatrix(Q), core.blrmatrix(R)

    return core.blrmatrix(Q), core.blrmatrix(R)
