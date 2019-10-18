# -*- coding: utf-8 -*-
import numpy
from ..core.mat import zmatrix
from ..core.mat import matrix
from ..core.mat import lrmatrix
from ..core.mat import blrmatrix


def qr(mat):
    """QR factorization."""
    if isinstance(mat, (matrix, numpy.ndarray)):
        q, r = numpy.linalg.qr(mat)
        return matrix(q), matrix(r)
    if isinstance(mat, blrmatrix):
        return _mbgs_for_blrmatrix(mat)
    return NotImplemented


def _tsqr_for_blrmatrix(blrmat):
    """Tall-Skinny QR factorization for BLR matrices.

    Arguments
    ---------
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
        if isinstance(X.A[i, 0], lrmatrix):
            Qi, Ri = numpy.linalg.qr(X.A[i, 0].U)
            Q[i] = Qi
            B[i] = Ri @ X.A[i, 0].V
        else:
            B[i] = X.A[i, 0]

    Qb, R = numpy.linalg.qr(numpy.vstack(B))
    rs, re = 0, 0

    for i in range(nb):
        if isinstance(X.A[i, 0], lrmatrix):
            rs, re = re, re + X.A[i, 0].rank
            Q[i] = lrmatrix((Q[i], Qb[rs:re, :]))
        else:
            rs, re = re, re + X.A[i, 0].shape[0]
            Q[i] = matrix(Qb[rs:re, :])

    return Q, matrix(R)


def _mbgs_for_blrmatrix(blrmat):
    """Modified block Gram-Schmidt algorithm for BLR matrices.

    Arguments
    ---------
    blrmat : blrmatrix
        A matrix to be factored.

    Returns
    -------
    Q : blrmatrix
        A matrix with orthonormal columns.
    R : blrmatrix
        The upper triangular matrix.
    """
    nb = min(blrmat.shape)
    X = blrmatrix(blrmat.A.copy())
    Q = numpy.full((X.shape[0], nb), None)
    R = numpy.full((nb, X.shape[1]), None)

    for index in numpy.ndindex(R.shape):
        rshape = X.A[index[0], index[0]].shape[1]
        cshape = X.A[index].shape[1]
        R[index] = zmatrix((rshape, cshape))

    for j in range(nb):
        Q[:, j], R[j, j] = _tsqr_for_blrmatrix(X[:, j])
        for k in range(j + 1, nb):
            Qj = blrmatrix(Q[:, j:j + 1])
            R[j, k] = (Qj.T @ X[:, k]).A[0, 0]
            X.A[:, k:k + 1] -= Qj.A @ R[j:j + 1, k:k + 1]

    return blrmatrix(Q), blrmatrix(R)
