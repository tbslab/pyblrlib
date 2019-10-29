# -*- coding: utf-8 -*-
import numbers
import numpy
from .. import core
from .. import linalg

class zmatrix(object):
    """A zero matrix object.

    Parameters
    ----------
    shape : tuple(int, int)
        A matrix shape.

    Attributes
    ----------
    T : matrix
        Transpose of self.
    shape : tuple(int, int)
        Shape of self.
    nbytes : int
        total bytes consumed by the elements of self.
    fnorm : float
        Frobenius norm of self.

    Examples
    --------
    Make the zmatrix instance.
    
    >>> import blrlib as bl
    >>> A = bl.zmatrix((2, 2))
    >>> print(A)
    zmatrix(2x2)
    """

    def __init__(self, shape):
        """Initialize self."""
        self._shape = shape

        if not (isinstance(shape, tuple) and len(shape) == 2):
            raise ValueError("'shape' must be tuple(int, int) object.")

    @property
    def T(self):
        """Return transpose of the zero matrix."""
        return zmatrix((self._shape[1], self._shape[0]))

    @property
    def shape(self):
        """Return shape of the zero matrix."""
        return self._shape

    @property
    def nbytes(self):
        """Return total bytes consumed by the elements of the zero matrix."""
        return 0

    @property
    def fnorm(self):
        """Return frobenius norm of the zero matrix."""
        return 0

    def __repr__(self):
        """Return the official string representation of this object."""
        return "zmatrix"

    def __str__(self):
        """Return the informal string representation of this object."""
        return "zmatrix({0[0]}x{0[1]})".format(self.shape)

    def __neg__(self):
        """Return -self"""
        return self

    def __pos__(self):
        """Return +self"""
        return self

    def __add__(self, other):
        """Return self + other."""
        if isinstance(other, zmatrix):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __sub__(self, other):
        """Return self - other."""
        if isinstance(other, zmatrix):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __mul__(self, other):
        """Return self * other."""
        if isinstance(other, numbers.Number):
            return self
        if isinstance(other, core.vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return core.vector(numpy.zeros((self.shape[0], 1)))
        if isinstance(other, zmatrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return zmatrix((self.shape[0], other.shape[1]))
        return NotImplemented

    def __rmul__(self, other):
        """Return other * self."""
        if isinstance(other, numbers.Number):
            return self
        if isinstance(other, core.vector):
            if other.shape[1] != self.shape[0]:
                raise ValueError("shape must be aligned")
            return core.vector(numpy.zeros((1, self.shape[1])))
        return NotImplemented

    def __matmul__(self, other):
        """Return self @ other."""
        if isinstance(other, core.vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return core.vector(numpy.zeros((self.shape[0], 1)))
        if isinstance(other, zmatrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return zmatrix((self.shape[0], other.shape[1]))
        return NotImplemented

    def __rmatmul__(self, other):
        """Return other @ self."""
        if isinstance(other, core.vector):
            if other.shape[1] != self.shape[0]:
                raise ValueError("shape must be aligned")
            return core.vector(numpy.zeros((1, self.shape[1])))
        return NotImplemented

    def __pow__(self, other):
        """Return self ** other."""
        if self.shape[0] != self.shape[1]:
            raise ValueError("shape must be square")
        return self

    def __rpow__(self, other):
        """Return other ** self."""
        if self.shape[0] != self.shape[1]:
            raise ValueError("shape must be square")
        return self

    def __eq__(self, other):
        """Return self == other."""
        if isinstance(other, zmatrix):
            return self.shape == other.shape
        return False

    def __ne__(self, other):
        """Return self != other."""
        return not (self == other)

    def __array__(self):
        """Return a reference to self."""
        return numpy.zeros(self._shape)


class matrix(object):
    """A matrix object utilizing a numpy.ndarray object.

    Parameters
    ----------
    obj : array like
        2-dimensional array object.

    Attributes
    ----------
    A : numpy.ndarray
        Self as numpy.ndarray object.
    T : matrix
        Transpose of self.
    I : matrix
        The (multiplicative) inverse of invertible self.
    shape : tuple(int, int)
        Shape of self.
    nbytes : int
        total bytes consumed by the elements of self.
    fnorm : float
        Frobenius norm of self.

    Examples
    --------
    Make the matrix instance.

    >>> import blrlib as bl
    >>> A = bl.matrix([[1, 2], [3, 4]])
    >>> A
    matrix
    [[1, 2],
    [3, 4]]
    """

    def __init__(self, obj):
        """Initialize self."""
        self._content = numpy.array(obj)

        if self._content.ndim != 2:
            raise ValueError("'matrix' must be 2-dimensional")
        if self._content.dtype not in (numpy.int, numpy.float, numpy.complex):
            raise TypeError("valid type instances must be set")

    @property
    def A(self):
        """Return self as numpy.ndarray object."""
        return self._content

    @property
    def T(self):
        """Return transpose of the matrix."""
        return matrix(self._content.T)

    @property
    def I(self):
        """Returns the (multiplicative) inverse of invertible self."""
        return matrix(numpy.linalg.inv(self._content))

    @property
    def shape(self):
        """Return shape of the matrix."""
        return self._content.shape

    @property
    def nbytes(self):
        """Return total bytes consumed by the elements of the matrix."""
        return self._content.nbytes

    @property
    def fnorm(self):
        """Return frobenius norm of the matrix."""
        return numpy.linalg.norm(self._content)

    def __repr__(self):
        """Return the official string representation of this object."""
        return "matrix\n" + str(self.A)

    def __str__(self):
        """Return the informal string representation of this object."""
        return "matrix({0[0]}x{0[1]})".format(self.shape)

    def __getitem__(self, key):
        """Return self[key]."""
        item = self.A[key]

        if item.ndim == 0:
            return matrix(item.reshape(1, 1))
        if item.ndim == 1:
            shape = item.shape[0]

            try:
                l = len(key)
            except Exception:
                l = 0

            if l > 1 and isinstance(key[1], int):
                return matrix(item.reshape(shape, 1))
            else:
                return matrix(item.reshape(1, shape))
        return matrix(item)

    def __setitem__(self, key, item):
        """Set item to self[key]."""
        self.A[key] = item

    def __neg__(self):
        """Return -self"""
        return matrix(-self.A)

    def __pos__(self):
        """Return +self"""
        return self

    def __add__(self, other):
        """Return self + other."""
        if isinstance(other, matrix):
            return matrix(self.A + other.A)
        if isinstance(other, zmatrix):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __radd__(self, other):
        """Return other + self."""
        if isinstance(other, zmatrix):
            if other.shape != self.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __sub__(self, other):
        """Return self - other."""
        if isinstance(other, matrix):
            return matrix(self.A - other.A)
        if isinstance(other, zmatrix):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __rsub__(self, other):
        """Return other - self."""
        if isinstance(other, zmatrix):
            if other.shape != self.shape:
                raise ValueError("shape must be aligned")
            return -self
        return NotImplemented

    def __mul__(self, other):
        """Return self * other."""
        if isinstance(other, numbers.Number):
            return matrix(self.A * other)
        if isinstance(other, core.vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return core.vector(self.A @ other.a)
        if isinstance(other, matrix):
            return matrix(self.A @ other.A)
        if isinstance(other, zmatrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return zmatrix((self.shape[0], other.shape[1]))
        return NotImplemented

    def __rmul__(self, other):
        """Return other * self."""
        if isinstance(other, numbers.Number):
            return matrix(other * self.A)
        if isinstance(other, core.vector):
            if other.shape[1] != self.shape[0]:
                raise ValueError("shape must be aligned")
            return core.vector(other.a @ self.A)
        if isinstance(other, zmatrix):
            if other.shape[1] != self.shape[0]:
                raise ValueError("shape must be aligned")
            return zmatrix((other.shape[0], self.shape[1]))
        return NotImplemented

    def __matmul__(self, other):
        """Return self @ other."""
        if isinstance(other, core.vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return core.vector(self.A @ other.a)
        if isinstance(other, matrix):
            return matrix(self.A @ other.A)
        if isinstance(other, zmatrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return zmatrix((self.shape[0], other.shape[1]))
        return NotImplemented

    def __rmatmul__(self, other):
        """Return other @ self."""
        if isinstance(other, core.vector):
            if other.shape[1] != self.shape[0]:
                raise ValueError("shape must be aligned")
            return core.vector(other.a @ self.A)
        if isinstance(other, zmatrix):
            if other.shape[1] != self.shape[0]:
                raise ValueError("shape must be aligned")
            return zmatrix((other.shape[0], self.shape[1]))
        return NotImplemented

    def __pow__(self, other):
        """Return self ** other."""
        return matrix(numpy.linalg.matrix_power(self.A, other))

    def __rpow__(self, other):
        """Return other ** self."""
        return matrix(numpy.linalg.matrix_power(self.A, other))

    def __eq__(self, other):
        """Return self == other."""
        if isinstance(other, matrix):
            return numpy.allclose(self.A, other.A)
        return False

    def __ne__(self, other):
        """Return self != other."""
        return not (self == other)

    def __array__(self):
        """Return a reference to self."""
        return self.A


class lrmatrix(object):
    """A low rank (LR) matrix object.

    Parameters
    ----------
    obj : array like or tuple(array like, array like)
        If you choose
            1. array like,
                you get the appriximation of this object.
            2. tuple(array like, array like),
                you get the LR matrix which have left matrix (tuple[0]) and
                right matrix (tuple[1]).
    mathod : str, optional
        A approximation method name. You can choose it from following list:
            1. svd
                Singular Value Decomposition Method.
            2. aca
                Adaptive Cross Approximation.
    eps : float, optional
        Numerical value for controlled accuracy.
    rank : int, optional
        Numerical rank for fixed rank approximation.

    Attributes
    ----------
    U : matrix
        Left matrix of self as matrix object.
    V : matrix
        Right matrix of self as matrix object.
    T : lrmatrix
        Transpose of self.
    shape : tuple(int, int)
        Shape of self.
    nbytes : int
        total bytes consumed by the elements of self.
    fnorm : float
        Frobenius norm of self.
    eps : float, optional
        Numerical value for controlled accuracy.
    rank : int, optional
        Numerical rank for fixed rank approximation.

    Examples
    --------
    Make the lrmatrix instance using SVD method.

    >>> import blrlib as bl
    >>> A = bl.matrix([[1, 2], [3, 4]])
    >>> X = bl.lrmatrix(A, mathod="svd", rank=1)
    >>> X
    lrmatrix
    left
    [[-2.21087956]
     [-4.99780755]]
    right
    [[-0.57604844 -0.81741556]]

    Make the lrmatrix instance.

    >>> import blrlib as bl
    >>> A = bl.matrix([[1], [2]])
    >>> B = bl.matrix([[3, 4]])
    >>> X = bl.lrmatrix((A, B))
    >>> X
    lrmatrix
    left
    [[1]
     [2]]
    right
    [[3, 4]]
    """

    def __init__(self, obj, method=None, eps=None, rank=None):
        """Initialize self."""
        self._eps = eps

        if isinstance(obj, tuple):
            self._left = matrix(obj[0])
            self._right = matrix(obj[1])

            if self._left.shape[1] != self._right.shape[0]:
                raise ValueError("shape must be aligned")
        elif method:
            if method.lower() == "svd":
                self._left, self._right = linalg.svda(obj, eps, rank)
            else:
                raise NotImplementedError("such method does not exist")
        else:
            raise ValueError("you must choose approximation method")

    @property
    def U(self):
        """Return left matrix as matrix object."""
        return self._left

    @property
    def V(self):
        """Return right matrix as matrix object."""
        return self._right

    @property
    def T(self):
        """Return transpose of the LR matrix."""
        return lrmatrix((self._right.T, self._left.T), eps=self.eps)

    @property
    def shape(self):
        """Return shape of the LR matrix."""
        return (self._left.shape[0], self._right.shape[1])

    @property
    def nbytes(self):
        """Return total bytes consumed by the elements of the LR matrix."""
        return self._left.nbytes + self._right.nbytes

    @property
    def fnorm(self):
        """Return frobenius norm of the LR matrix."""
        return numpy.linalg.norm(self._left @ self._right)

    @property
    def rank(self):
        """Return numerical rank of the LR matrix."""
        return self._left.shape[1]

    @property
    def eps(self):
        """Return numerical value for controlling approximation accuracy."""
        return self._eps

    def __repr__(self):
        """Return the official string representation of this object."""
        return "lrmatrix\nleft\n" + str(self.U.A) + "\nright\n" + str(self.V.A)

    def __str__(self):
        """Return the informal string representation of this object."""
        return "lrmatrix({0[0]}x{0[1]}, {1})".format(self.shape, self.rank)

    def __neg__(self):
        """Return -self."""
        return lrmatrix((-self.U, self.V), eps=self.eps)

    def __pos__(self):
        """Return +self."""
        return self

    def __add__(self, other):
        """Return self + other."""
        if isinstance(other, matrix):
            return matrix(self.U @ self.V + other)
        if isinstance(other, lrmatrix):
            return self._rounded_addition(other)
        if isinstance(other, zmatrix):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __radd__(self, other):
        """Return other + self."""
        if isinstance(other, matrix):
            return matrix(other + self.U @ self.V)
        if isinstance(other, zmatrix):
            if other.shape != self.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def _rounded_addition(self, other):
        """Return result of rounded addition for two LR matrices."""
        Qu, Ru = numpy.linalg.qr(numpy.hstack([self.U, other.U]))
        Qv, Rv = numpy.linalg.qr(numpy.hstack([self.V.T, other.V.T]))
        U, s, Vh = numpy.linalg.svd(Ru @ Rv.T)

        if self.eps:
            accuracy_bound = self.eps * numpy.linalg.norm(s)
            new_rank = 1

            while numpy.linalg.norm(s[new_rank:]) >= accuracy_bound:
                new_rank += 1

            left = (Qu @ U)[:, :new_rank] * s[:new_rank]
            right = (Vh @ Qv.T)[:new_rank, :]
            return lrmatrix((left, right), eps=self.eps)

        left = (Qu @ U)[:, :self.rank] * s[:self.rank]
        right = (Vh @ Qv.T)[:self.rank, :]
        return lrmatrix((left, right))

    def __sub__(self, other):
        """Return self - other."""
        if isinstance(other, matrix):
            return matrix(self.U @ self.V - other.A)
        if isinstance(other, lrmatrix):
            return self._rounded_addition(-other)
        if isinstance(other, zmatrix):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __rsub__(self, other):
        """Return other - self."""
        if isinstance(other, matrix):
            return matrix(other.A - self.U @ self.V)
        if isinstance(other, zmatrix):
            if other.shape != self.shape:
                raise ValueError("shape must be aligned")
            return -self
        return NotImplemented

    def __mul__(self, other):
        """Return self * other."""
        if isinstance(other, numbers.Number):
            return lrmatrix((self.U, self.V * other), eps=self.eps)
        if isinstance(other, core.vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return core.vector(self.U @ (self.V @ other))
        if isinstance(other, matrix):
            return self @ other
        if isinstance(other, lrmatrix):
            return self @ other
        if isinstance(other, zmatrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return zmatrix((self.shape[0], other.shape[1]))
        return NotImplemented

    def __rmul__(self, other):
        """Return other * self."""
        if isinstance(other, numbers.Number):
            return lrmatrix((other * self.U, self.V), eps=self.eps)
        if isinstance(other, core.vector):
            if other.shape[1] != self.shape[0]:
                raise ValueError("shape must be aligned")
            return core.vector((other @ self.U) @ self.V)
        if isinstance(other, matrix):
            return other @ self
        if isinstance(other, zmatrix):
            if other.shape[1] != self.shape[0]:
                raise ValueError("shape must be aligned")
            return zmatrix((other.shape[0], self.shape[1]))
        return NotImplemented

    def __matmul__(self, other):
        """Return self @ other."""
        if isinstance(other, core.vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return core.vector(self.U @ (self.V @ other))
        if isinstance(other, matrix):
            return lrmatrix((self.U, self.V @ other), eps=self.eps)
        if isinstance(other, lrmatrix):
            X = self.V @ other.U
            if self.rank <= other.rank:
                return lrmatrix((self.U, X @ other.V), eps=self.eps)
            return lrmatrix((self.U @ X, other.V), eps=self.eps)
        if isinstance(other, zmatrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return zmatrix((self.shape[0], other.shape[1]))
        return NotImplemented

    def __rmatmul__(self, other):
        """Return other @ self."""
        if isinstance(other, core.vector):
            if other.shape[1] != self.shape[0]:
                raise ValueError("shape must be aligned")
            return core.vector((other @ self.U) @ self.V)
        if isinstance(other, matrix):
            return lrmatrix((other @ self.U, self.V), eps=self.eps)
        if isinstance(other, zmatrix):
            if other.shape[1] != self.shape[0]:
                raise ValueError("shape must be aligned")
            return zmatrix((other.shape[0], self.shape[1]))
        return NotImplemented

    def __pow__(self, other):
        """Return self ** other."""
        X = (self.V @ self.U) ** (other - 1)
        return lrmatrix((self.U, X @ self.V), eps=self.eps)

    def __rpow__(self, other):
        """Return other ** self."""
        X = (other - 1) ** (self.V @ self.U)
        return lrmatrix((self.U @ X, self.V), eps=self.eps)

    def __eq__(self, other):
        """Return self == other."""
        if isinstance(other, lrmatrix):
            return (self.U == other.U) and (self.V == other.V)
        return False

    def __ne__(self, other):
        """Return self != other."""
        return not (self == other)

    def __array__(self):
        """Return a reference to self."""
        return (self.U @ self.V).__array__()


class blrmatrix(object):
    """A block low rank (BLR) matrix object.

    Parameters
    ----------
    obj : numpy.ndarray
        2-dimensional numpy.ndarray. Each element must be either matrix or
        lrmatrix object.

    Attributes
    ----------
    A : numpy.ndarray
        Self as numpy.ndarray object.
    T : blrmatrix
        Transpose of self.
    I : blrmatrix
        The (multiplicative) inverse of invertible self.
    shape : tuple(int, int)
        Shape of self.
    nbytes : int
        total bytes consumed by the elements of self.
    fnorm : float
        Frobenius norm of self.

    See also
    --------
    build_blrmatrix : Generate a blrmatrix instance.
    """

    def __init__(self, obj):
        """Initialize self."""
        self._block = numpy.array(obj, dtype=object)

        if self._block.ndim != 2:
            raise ValueError("'blrmatrix' must be 2-dimensional")
        for index in numpy.ndindex(self._block.shape):
            if not isinstance(self._block[index], (matrix, lrmatrix, zmatrix)):
                raise TypeError("valid type instances must be set")

    @property
    def A(self):
        """Return self as numpy.ndarray object."""
        return self._block

    @property
    def T(self):
        """Return transpose of the BLR matrix."""
        block = numpy.full(self._block.T.shape, None)

        for index in numpy.ndindex(self._block.shape):
            block[index[1], index[0]] = self._block[index].T
        return blrmatrix(block)

    @property
    def I(self):
        """Returns the (multiplicative) inverse of invertible self."""
        return NotImplemented

    @property
    def shape(self):
        """Return shape of the BLR matrix"""
        return self._block.shape

    @property
    def nbytes(self):
        """Return total bytes consumed by the elements of the BLR matrix."""
        return sum(self._block[index].nbytes
                   for index in numpy.ndindex(self._block.shape))

    @property
    def fnorm(self):
        """Return frobenius norm of the BLR matrix."""
        return numpy.sqrt(sum(self._block[index].fnorm ** 2
                              for index in numpy.ndindex(self._block.shape)))

    def __repr__(self):
        """Return the official string representation of this object."""
        out = "blrmatrix"

        for index in numpy.ndindex(self.shape):
            out += "\n{0}: {1}".format(index, repr(self.A[index]))
        return out

    def __str__(self):
        """Return the informal string representation of this object."""
        out = "blrmatrix({0[0]}x{0[1]})".format(self.shape)

        for index in numpy.ndindex(self.shape):
            out += "\n{0}: {1}".format(index, str(self.A[index]))
        return out

    def __getitem__(self, key):
        """Return self[key]."""
        item = self.A[key]

        if isinstance(item, (matrix, lrmatrix)):
            return blrmatrix([[item]])
        if item.ndim == 1:
            shape = item.shape[0]

            try:
                l = len(key)
            except Exception:
                l = 0

            if l > 1 and isinstance(key[1], int):
                return blrmatrix(item.reshape(shape, 1))
            else:
                return blrmatrix(item.reshape(1, shape))
        return blrmatrix(item)

    def __neg__(self):
        """Return -self."""
        return blrmatrix(-self.A)

    def __pos__(self):
        """Return +self."""
        return self

    def __add__(self, other):
        """Return self + other."""
        if isinstance(other, blrmatrix):
            return blrmatrix(self.A + other.A)
        return NotImplemented

    def __radd__(self, other):
        """Return other + self."""
        return NotImplemented

    def __sub__(self, other):
        """Return self - other."""
        if isinstance(other, blrmatrix):
            return blrmatrix(self.A - other.A)
        return NotImplemented

    def __rsub__(self, other):
        """Return other - self"""
        return NotImplemented

    def __mul__(self, other):
        """Return self * other."""
        if isinstance(other, numbers.Number):
            return self.A * other
        if isinstance(other, core.vector):
            return self._mul_by_vector(other)
        if isinstance(other, blrmatrix):
            return blrmatrix(self.A @ other.A)
        return NotImplemented

    def _mul_by_vector(self, other):
        """Return self * other (vector object)."""
        return NotImplemented

    def __rmul__(self, other):
        """Return other * self."""
        if isinstance(other, numbers.Number):
            return other * self.A
        if isinstance(other, core.vector):
            return self._rmul_by_vector(other)
        return NotImplemented

    def _rmul_by_vector(self, other):
        """Return other (vector object) * self."""
        return NotImplemented

    def __matmul__(self, other):
        """Return self @ other."""
        if isinstance(other, core.vector):
            return self._mul_by_vector(other)
        if isinstance(other, blrmatrix):
            return blrmatrix(self.A @ other.A)
        return NotImplemented

    def __rmatmul__(self, other):
        """Return other @ self."""
        if isinstance(other, core.vector):
            return self._rmul_by_vector(other)
        return NotImplemented

    def __pow__(self, other):
        """Return self ** other."""
        return blrmatrix(numpy.linalg.matrix_power(self.A, other))

    def __rpow__(self, other):
        """Return other ** self."""
        return blrmatrix(numpy.linalg.matrix_power(self.A, other))

    def __eq__(self, other):
        """Return self == other."""
        if isinstance(other, blrmatrix):
            return (self.A == other.A).all()
        return False

    def __ne__(self, other):
        """Return self != other."""
        return not (self == other)

    def __array__(self):
        """Return a reference to self."""
        return self.A

    def to_matrix(self):
        """Return self as matrix object."""
        return matrix(numpy.block(self.A.tolist()))


def build_blrmatrix(mat, structure, indices=None, method=None, eps=None, rank=None):
    """Return block low rank (BLR) Matrix.

    Parameters
    ----------
    mat : array like
        2-dimensional array object.
    structure : int, list or tuple of row and column block size
        This decide a structure of BLR matrix. If you choose
            1. int,
                you get a regular block structure.
            2. list or tuple of row and column block size,
                you get a irregular block structure as you specified.
    indices : list or tuple of tuple(row index, column index), optional
        This is a block index list which specifies which block should be matrix
        object. If you do not give this parameter, this function will generate
        normal block matrix, although that instance is blrmatrix object.
    mathod : str, optionnal
        A approximation method name. You can choose it from following list:
            1. svd
                Singular Value Decomposition Method.
            2. aca
                Adaptive Cross Approximation.
    eps : float, optional
        Numerical value for controlled accuracy.
    rank : int, optional
        Numerical rank for fixed rank approximation.

    Returns
    -------
        blrmatrix :
            BLR matrix sutisfying the conditions you gave.

    Examples
    --------
    You can generate the BLR matrix which has LR matrices
    at non-diagonals as following.

    >>> import numpy as np
    >>> import blrlib as bl
    >>> A = np.random.randint(1, 5, (100, 100))
    >>> nb = 10
    >>> indices = [(i, i) for i in range(nb)]
    >>> X = bl.build_blrmatrix(A, nb, indices, method="svd", rank=1)
    >>> print(X)
    blrmatrix(10x10)
    (0, 0): matrix(10x10)
    (0, 1): lrmatrix(10x10, 1)
    (0, 2): lrmatrix(10x10, 1)
    ...
    (9, 9): matrix(10x10)

    You can generate BLR matrix which has a irregular block structure as
    following. In practice, you don't need to fix sizes like structure[0] and
    structure[1] are to be same size, but you need to fix sizes like these twos
    are compatible with obj.shape.

    >>> import numpy as np
    >>> import blrlib as bl
    >>> A = np.random.randint(1, 5, (100, 100))
    >>> nb = 8
    >>> indices = ((i, i) for i in range(nb))
    >>> structure = [
        [5, 15, 20, 15, 10, 25, 5, 5], # row block shapes
        [15, 25, 10, 5, 20, 5, 15, 5]  # column block shapes
    ]
    >>> X = bl.build_blrmatrix(A, structure, indices, method="svd", eps=1e-4)
    >>> print(X)
    blrmatrix(8x8)
    (0, 0): matrix(5x15)
    (0, 1): lrmatrix(5x25, 1)
    (0, 2): lrmatrix(5x10, 1)
    (0, 3): lrmatrix(5x5, 1)
    ...
    (7, 7): matrix(5x5)
    """
    if not isinstance(mat, numpy.ndarray):
        mat = numpy.array(mat)
    if mat.ndim != 2:
        raise ValueError("'mat' must be 2-dimensional")
    if isinstance(structure, int):
        nb = structure
        if nb < 1:
            raise ValueError("'structure' must be larger than 1")
        rshapes = [mat.shape[0] // nb for _ in range(nb - 1)]
        rshapes.append(mat.shape[0] // nb + mat.shape[0] % nb)
        cshapes = [mat.shape[1] // nb for _ in range(nb - 1)]
        cshapes.append(mat.shape[1] // nb + mat.shape[1] % nb)
        rshapes = numpy.array(rshapes, dtype=int)
        cshapes = numpy.array(cshapes, dtype=int)
    if isinstance(structure, (list, tuple)):
        if len(structure) != 2:
            raise ValueError("'structure' must has two sequences.")
        rshapes = numpy.array(structure[0], dtype=int)
        cshapes = numpy.array(structure[1], dtype=int)
        if not rshapes.ndim == cshapes.ndim == 1:
            raise ValueError("'structure' elements must be 1-dimensional")
        if not mat.shape == (rshapes.sum(), cshapes.sum()):
            raise ValueError("'structure' must be compatible with 'mat.shape'")
    if indices:
        if not isinstance(indices, (list, tuple)):
            raise ValueError("'indices' must be a list or tuple")
        if not indices:
            raise ValueError("list or tuple must not be empty")
        if not isinstance(indices[0], tuple):
            raise ValueError("'indices' elements must be tuple(row, column)")
    else:
        indices = ()

    block = numpy.full((rshapes.size, cshapes.size), None)

    for index in numpy.ndindex(block.shape):
        i, j = index
        rs, re = rshapes[:i].sum(), rshapes[:i + 1].sum()
        cs, ce = cshapes[:j].sum(), cshapes[:j + 1].sum()

        if index in indices:
            block[index] = matrix(mat[rs:re, cs:ce])
        else:
            block[index] = lrmatrix(mat[rs:re, cs:ce], method, eps, rank)

    return blrmatrix(block)
