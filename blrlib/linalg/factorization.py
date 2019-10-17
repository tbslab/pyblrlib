# -*- coding: utf-8 -*-
import numpy
from blrlib.core.mat import zmatrix, matrix, lrmatrix, blrmatrix


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
    Q = numpy.full(blrmat.shape[0], None)
    B = []

    for i in range(blrmat.shape[0]):
        if isinstance(blrmat.A[i, 0], lrmatrix):
            Qi, Ri = numpy.linalg.qr(blrmat.A[i, 0].U)
            Q[i] = Qi
            B.append(Ri @ blrmat.A[i, 0].V)
        else:
            B.append(blrmat.A[i, 0])

    Qb, R = numpy.linalg.qr(numpy.vstack(B))
    rs, re = 0, 0

    for i in range(blrmat.shape[0]):
        if isinstance(blrmat.A[i, 0], lrmatrix):
            rs, re = re, re + blrmat.A[i, 0].rank
            Q[i] = lrmatrix((Q[i], Qb[rs:re, :]))
        else:
            rs, re = re, re + blrmat.A[i, 0].shape[0]
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
    _blrmat = blrmatrix(blrmat.A.copy())
    min_nb = min(_blrmat.shape)
    Q = numpy.full((_blrmat.shape[0], min_nb), None)
    R = numpy.full((min_nb, _blrmat.shape[1]), None)
    for index in numpy.ndindex(R.shape):
        R[index] = zmatrix(_blrmat.A[index].shape)

    for j in range(min_nb):
        Q[:, j], R[j, j] = _tsqr_for_blrmatrix(_blrmat[:, j])
        for k in range(j + 1, min_nb):
            Qj = blrmatrix(Q[:, j:j + 1])
            R[j, k] = (Qj.T @ _blrmat[:, k]).A[0, 0]
            _blrmat.A[:, k:k + 1] -= Qj.A @ R[j:j + 1, k:k + 1]

    return blrmatrix(Q), blrmatrix(R)
