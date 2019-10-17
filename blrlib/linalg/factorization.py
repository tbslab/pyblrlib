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
        A matrix with orthonormal columns. The list of matrix or lrmatrix
        objects.
    R : numpy.ndarray
        The upper triangular matrix.
    """
    nb = blrmat.shape[0]
    Q = numpy.full(nb, None)
    B = []

    for i in range(nb):
        if isinstance(blrmat.A[i, 0], lrmatrix):
            Qi, Ri = numpy.linalg.qr(blrmat.A[i, 0].U)
            Q[i] = Qi
            B.append(Ri @ blrmat.A[i, 0].V)
        else:
            B.append(blrmat.A[i, 0])

    Qb, R = numpy.linalg.qr(numpy.vstack(B))
    rs, re = 0, 0

    for i in range(nb):
        if isinstance(blrmat.A[i, 0], lrmatrix):
            rs, re = re, re + blrmat.A[i, 0].rank
            Q[i] = lrmatrix((Q[i], Qb[rs:re, :]))
        else:
            rs, re = re, re + blrmat.A[i, 0].shape[0]
            Q[i] = matrix(Qb[rs:re, :])

    return Q, R


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
    ...
