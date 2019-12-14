# -*- coding: utf-8 -*-
import numbers
import numpy
from .. import core
from .. import linalg


class Zero(object):
    """A zero matrix object.

    Attributes
    ----------
    T: Zero
        Transpose of self.
    shape: tuple of int
        Shape of self.
    nbytes: int
        total bytes consumed by the elements of self.

    Examples
    --------
    Make the Zero instance.

    >>> import blrlib as bl
    >>> A = bl.Zero((2, 2))
    >>> print(A)
    Zero(2x2)
    """

    def __init__(self, shape):
        """Initialize self.

        Parameters
        ----------
        shape: tuple of int
            A matrix shape.
        """
        self._shape = shape

        if not (isinstance(shape, tuple) and len(shape) == 2):
            raise ValueError("'shape' must be tuple(int, int) object.")

    @property
    def T(self):
        """Return transpose of the zero matrix."""
        return Zero((self._shape[1], self._shape[0]))

    @property
    def shape(self):
        """Return shape of the zero matrix."""
        return self._shape

    @property
    def nbytes(self):
        """Return total bytes consumed by the elements of the zero matrix."""
        return 0

    def __repr__(self):
        """Return the official string representation of this object."""
        return "Zero"

    def __str__(self):
        """Return the informal string representation of this object."""
        return "Zero({0[0]}x{0[1]})".format(self.shape)

    def __neg__(self):
        """Return -self"""
        return self

    def __pos__(self):
        """Return +self"""
        return self

    def __add__(self, other):
        """Return self + other."""
        if isinstance(other, Zero):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __sub__(self, other):
        """Return self - other."""
        if isinstance(other, Zero):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __mul__(self, other):
        """Return self * other."""
        if isinstance(other, numbers.Number):
            return self
        if isinstance(other, core.Vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return core.Vector(numpy.zeros((self.shape[0], 1)))
        if isinstance(other, Zero):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return Zero((self.shape[0], other.shape[1]))
        return NotImplemented

    def __rmul__(self, other):
        """Return other * self."""
        if isinstance(other, numbers.Number):
            return self
        if isinstance(other, core.Vector):
            if other.shape[1] != self.shape[0]:
                raise ValueError("shape must be aligned")
            return core.Vector(numpy.zeros((1, self.shape[1])))
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
        if isinstance(other, Zero):
            return self.shape == other.shape
        return False

    def __ne__(self, other):
        """Return self != other."""
        return not (self == other)

    def __array__(self):
        """Return a reference to self."""
        return numpy.zeros(self.shape)


class Dense(object):
    """A dense matrix object utilizing a numpy.ndarray object.

    Attributes
    ----------
    T: Dense
        Transpose of self.
    I: Dense
        The (multiplicative) inverse of invertible self.
    shape: tuple of int
        Shape of self.
    nbytes: int
        total bytes consumed by the elements of self.

    Examples
    --------
    Make the Dense instance.

    >>> import blrlib as bl
    >>> A = bl.Dense([[1, 2], [3, 4]])
    >>> A
    Dense
    [[1, 2],
     [3, 4]]
    """

    def __init__(self, obj):
        """Initialize self.

        Parameters
        ----------
        obj: array_like
            2 dimensional array object.
        """
        self._m = numpy.array(obj)

        if self._m.ndim != 2:
            raise ValueError("'Dense' must be 2 dimensional")
        if self._m.dtype not in (numpy.int, numpy.float):
            raise TypeError("valid type instances must be set")

    @property
    def T(self):
        """Return transpose of the matrix."""
        return Dense(self._m.T)

    @property
    def I(self):
        """Returns the (multiplicative) inverse of invertible self."""
        return Dense(numpy.linalg.inv(self._m))

    @property
    def shape(self):
        """Return shape of the matrix."""
        return self._m.shape

    @property
    def nbytes(self):
        """Return total bytes consumed by the elements of the matrix."""
        return self._m.nbytes

    def __repr__(self):
        """Return the official string representation of this object."""
        return "Dense\n" + str(self._m)

    def __str__(self):
        """Return the informal string representation of this object."""
        return "Dense({0[0]}x{0[1]})".format(self.shape)

    def __getitem__(self, key):
        """Return self[key]."""
        item = self._m[key]

        if item.ndim == 0:
            return Dense(item.reshape(1, 1))
        if item.ndim == 1:
            shape = item.shape[0]

            try:
                l = len(key)
            except Exception:
                l = 0

            if l > 1 and isinstance(key[1], int):
                return Dense(item.reshape(shape, 1))
            else:
                return Dense(item.reshape(1, shape))
        return Dense(item)

    def __setitem__(self, key, item):
        """Set item to self[key]."""
        if isinstance(item, Dense):
            if item.shape[1] != 1:
                self._m[key] = item._m.tolist()
            self._m[key] = item._m.flatten()
        else:
            self._m[key] = item

    def __neg__(self):
        """Return -self"""
        return Dense(-self._m)

    def __pos__(self):
        """Return +self"""
        return self

    def __add__(self, other):
        """Return self + other."""
        if isinstance(other, Dense):
            return Dense(self._m + other._m)
        if isinstance(other, Zero):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __radd__(self, other):
        """Return other + self."""
        if isinstance(other, Zero):
            if other.shape != self.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __sub__(self, other):
        """Return self - other."""
        if isinstance(other, Dense):
            return Dense(self._m - other._m)
        if isinstance(other, Zero):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __rsub__(self, other):
        """Return other - self."""
        if isinstance(other, Zero):
            if other.shape != self.shape:
                raise ValueError("shape must be aligned")
            return -self
        return NotImplemented

    def __mul__(self, other):
        """Return self * other."""
        if isinstance(other, numbers.Number):
            return Dense(self._m * other)
        if isinstance(other, core.Vector):
            return core.Vector(numpy.dot(self._m, other.a))
        if isinstance(other, Dense):
            return Dense(numpy.matmul(self._m, other._m))
        if isinstance(other, Zero):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return Zero((self.shape[0], other.shape[1]))
        return NotImplemented

    def __rmul__(self, other):
        """Return other * self."""
        if isinstance(other, numbers.Number):
            return Dense(other * self._m)
        if isinstance(other, core.Vector):
            return core.Vector(numpy.dot(other.a, self._m))
        if isinstance(other, Zero):
            if other.shape[1] != self.shape[0]:
                raise ValueError("shape must be aligned")
            return Zero((other.shape[0], self.shape[1]))
        return NotImplemented

    def __pow__(self, other):
        """Return self ** other."""
        return Dense(numpy.linalg.matrix_power(self._m, other))

    def __rpow__(self, other):
        """Return other ** self."""
        return Dense(numpy.linalg.matrix_power(self._m, other))

    def __eq__(self, other):
        """Return self == other."""
        if isinstance(other, Dense):
            return numpy.allclose(self._m, other._m)
        return False

    def __ne__(self, other):
        """Return self != other."""
        return not (self == other)

    def __array__(self):
        """Return a reference to self."""
        return self._m


class LowRank(object):
    """A low rank (LR) matrix object.

    Attributes
    ----------
    U: Dense
        Left matrix of self as Dense object.
    V: Dense
        Right matrix of self as Dense object.
    T: LowRank
        Transpose of self.
    shape: tuple of int
        Shape of self.
    nbytes: int
        total bytes consumed by the elements of self.
    eps: float or None
        Numerical value for controlled accuracy.
    rank: int
        Numerical rank for fixed rank approximation.
    method: str
        A method name for low rank approximation.

    Examples
    --------
    Make the LowRank instance using SVD method.

    >>> import blrlib as bl
    >>> A = bl.Dense([[1, 2], [3, 4]])
    >>> X = bl.LowRank(A, mathod="svd", rank=1)
    >>> X
    LowRank
    left
    [[-2.21087956]
     [-4.99780755]]
    right
    [[-0.57604844 -0.81741556]]
    """

    def __init__(self, obj, method="svd", eps=None, rank=None):
        """Initialize self.

        Parameters
        ----------
        obj: array_like or tuple of array_like
            If you choose,

            1. array_like
                you get the appriximation of this object.
            2. tuple of array_like
                you get the LR matrix which have left matrix:tuple[0] and
                right matrix:tuple[1].
        method: str, default 'svd'
            A method name for low rank approximation. You can choose it
            from following list.

            1. 'svd'
                Singular Value Decomposition Method.
            2. 'aca'
                Adaptive Cross Approximation.
        eps: float, default None
            Numerical value for controlled accuracy.
        rank: int, default None
            Numerical rank for fixed rank approximation.
        """
        self._eps = eps
        self._method = method

        if isinstance(obj, tuple):
            self._lm = Dense(obj[0])
            self._rm = Dense(obj[1])

            if self._lm.shape[1] != self._rm.shape[0]:
                raise ValueError("shape must be aligned")
        elif method:
            if method.lower() == "svd":
                self._lm, self._rm = linalg.truncated_svd(obj, eps, rank)
            elif method.lower() == "aca":
                self._lm, self._rm = linalg.aca(obj, eps, rank)
            else:
                raise NotImplementedError("such method does not exist")

    @property
    def U(self):
        """Return left matrix as Dense object."""
        return self._lm

    @property
    def V(self):
        """Return right matrix as Dense object."""
        return self._rm

    @property
    def T(self):
        """Return transpose of the LR matrix."""
        return LowRank((self._rm.T, self._lm.T), self._method, self._eps)

    @property
    def shape(self):
        """Return shape of the LR matrix."""
        return (self._lm.shape[0], self._rm.shape[1])

    @property
    def nbytes(self):
        """Return total bytes consumed by the elements of the LR matrix."""
        return self._lm.nbytes + self._rm.nbytes

    @property
    def rank(self):
        """Return numerical rank of the LR matrix."""
        return self._lm.shape[1]

    @property
    def eps(self):
        """Return numerical value for controlling approximation accuracy."""
        return self._eps

    @property
    def method(self):
        """Return method name for low rank approximation."""
        return self._method

    def __repr__(self):
        """Return the official string representation of this object."""
        return "LowRank\nleft\n" + str(self.U._m) + "\nright\n" + str(self.V._m)

    def __str__(self):
        """Return the informal string representation of this object."""
        return "LowRank({0[0]}x{0[1]}, {1})".format(self.shape, self.rank)

    def __neg__(self):
        """Return -self."""
        return LowRank((-self.U, self.V), self.method, self.eps)

    def __pos__(self):
        """Return +self."""
        return self

    def __add__(self, other):
        """Return self + other."""
        if isinstance(other, Dense):
            return Dense(self.U * self.V + other)
        if isinstance(other, LowRank):
            return self._formatted_add(other)
        if isinstance(other, Zero):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __radd__(self, other):
        """Return other + self."""
        if isinstance(other, Dense):
            return Dense(other + self.U * self.V)
        if isinstance(other, Zero):
            if other.shape != self.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def _formatted_add(self, other):
        """Return result of formatted addition for two LR matrices."""
        Qu, Ru = numpy.linalg.qr(numpy.hstack([self.U, other.U]))
        Qv, Rv = numpy.linalg.qr(numpy.hstack([self.V.T, other.V.T]))
        U, s, Vh = numpy.linalg.svd(numpy.matmul(Ru, Rv.T))

        if self.eps:
            rank = 1

            while s[rank] >= self.eps:
                rank += 1

            L = numpy.matmul(Qu, U)[:, :rank] * s[:rank]
            R = numpy.matmul(Vh, Qv.T)[:rank, :]
            return LowRank((L, R), self.method, self.eps)

        L = numpy.matmul(Qu, U)[:, :self.rank] * s[:self.rank]
        R = numpy.matmul(Vh, Qv.T)[:self.rank, :]
        return LowRank((L, R), self.method)

    def __sub__(self, other):
        """Return self - other."""
        if isinstance(other, Dense):
            return Dense(self.U * self.V - other)
        if isinstance(other, LowRank):
            return self._formatted_add(-other)
        if isinstance(other, Zero):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return self
        return NotImplemented

    def __rsub__(self, other):
        """Return other - self."""
        if isinstance(other, Dense):
            return Dense(other - self.U * self.V)
        if isinstance(other, Zero):
            if other.shape != self.shape:
                raise ValueError("shape must be aligned")
            return -self
        return NotImplemented

    def __mul__(self, other):
        """Return self * other."""
        if isinstance(other, numbers.Number):
            return LowRank((self.U, self.V * other), self.method, self.eps)
        if isinstance(other, core.Vector):
            return core.Vector(self.U * (self.V * other))
        if isinstance(other, Dense):
            return LowRank((self.U, self.V * other), self.method, self.eps)
        if isinstance(other, LowRank):
            X = self.V * other.U
            if self.rank <= other.rank:
                return LowRank((self.U, X * other.V), self.method, self.eps)
            return LowRank((self.U * X, other.V), self.method, self.eps)
        if isinstance(other, Zero):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            return Zero((self.shape[0], other.shape[1]))
        return NotImplemented

    def __rmul__(self, other):
        """Return other * self."""
        if isinstance(other, numbers.Number):
            return LowRank((other * self.U, self.V), self.method, self.eps)
        if isinstance(other, core.Vector):
            return core.Vector((other * self.U) * self.V)
        if isinstance(other, Dense):
            return LowRank((other * self.U, self.V), self.method, self.eps)
        if isinstance(other, Zero):
            if other.shape[1] != self.shape[0]:
                raise ValueError("shape must be aligned")
            return Zero((other.shape[0], self.shape[1]))
        return NotImplemented

    def __pow__(self, other):
        """Return self ** other."""
        X = (self.V * self.U) ** (other - 1)
        return LowRank((self.U, X * self.V), self.method, self.eps)

    def __rpow__(self, other):
        """Return other ** self."""
        X = (other - 1) ** (self.V * self.U)
        return LowRank((self.U * X, self.V), self.method, self.eps)

    def __eq__(self, other):
        """Return self == other."""
        if isinstance(other, LowRank):
            return (self.U == other.U) and (self.V == other.V)
        return False

    def __ne__(self, other):
        """Return self != other."""
        return not (self == other)

    def __array__(self):
        """Return a reference to self."""
        return (self.U * self.V).__array__()


class BlockLowRank(object):
    """A block low rank (BLR) matrix object.

    Attributes
    ----------
    T: BlockLowRank
        Transpose of self.
    I: BlockLowRank
        The (multiplicative) inverse of invertible self.
    shape: tuple of int
        Shape of self.
    nbytes: int
        total bytes consumed by the elements of self.

    See also
    --------
    build_blrmatrix: Generate a BlockLowRank instance.
    """

    def __init__(self, obj):
        """Initialize self.

        Parameters
        ----------
        obj: array_like
            2 dimensional array object. Each element must be either
            Dense, LowRank or Zero object.
        """
        self._b = numpy.array(obj, dtype=object)

        if self._b.ndim != 2:
            raise ValueError("'BlockLowRank' must be 2 dimensional")
        for index in numpy.ndindex(self._b.shape):
            if not isinstance(self._b[index], (Dense, LowRank, Zero)):
                raise TypeError("valid type instances must be set")

    @property
    def T(self):
        """Return transpose of the BLR matrix."""
        B = numpy.full(self._b.T.shape, None)

        for i, j in numpy.ndindex(self._b.shape):
            B[j, i] = self._b[i, j].T
        return BlockLowRank(B)

    @property
    def I(self):
        """Returns the (multiplicative) inverse of invertible self."""
        return NotImplemented

    @property
    def shape(self):
        """Return shape of the BLR matrix"""
        return self._b.shape

    @property
    def nbytes(self):
        """Return total bytes consumed by the elements of the BLR matrix."""
        return sum(self._b[index].nbytes for index in numpy.ndindex(self._b.shape))

    def __repr__(self):
        """Return the official string representation of this object."""
        out = "BlockLowRank"

        for index in numpy.ndindex(self.shape):
            out += "\n{0}: {1}".format(index, repr(self._b[index]))
        return out

    def __str__(self):
        """Return the informal string representation of this object."""
        out = "BlockLowRank({0[0]}x{0[1]})".format(self.shape)

        for index in numpy.ndindex(self.shape):
            out += "\n{0}: {1}".format(index, str(self._b[index]))
        return out

    def __getitem__(self, key):
        """Return self[key]."""
        item = self._b[key]

        if isinstance(item, (Dense, LowRank, Zero)):
            return BlockLowRank([[item]])
        if item.ndim == 1:
            shape = item.shape[0]

            try:
                l = len(key)
            except Exception:
                l = 0

            if l > 1 and isinstance(key[1], int):
                return BlockLowRank(item.reshape(shape, 1))
            else:
                return BlockLowRank(item.reshape(1, shape))
        return BlockLowRank(item)

    def __setitem__(self, key, item):
        """Set item to self[key]."""
        if isinstance(item, BlockLowRank):
            if item.shape[1] != 1:
                self._b[key] = item._b.tolist()
            self._b[key] = item._b.flatten()
        else:
            self._b[key] = item

    def __neg__(self):
        """Return -self."""
        return BlockLowRank(-self._b)

    def __pos__(self):
        """Return +self."""
        return self

    def __add__(self, other):
        """Return self + other."""
        if isinstance(other, BlockLowRank):
            return BlockLowRank(self._b + other._b)
        return NotImplemented

    def __radd__(self, other):
        """Return other + self."""
        return NotImplemented

    def __sub__(self, other):
        """Return self - other."""
        if isinstance(other, BlockLowRank):
            return BlockLowRank(self._b - other._b)
        return NotImplemented

    def __rsub__(self, other):
        """Return other - self"""
        return NotImplemented

    def __mul__(self, other):
        """Return self * other."""
        if isinstance(other, numbers.Number):
            return self._b * other
        if isinstance(other, core.Vector):
            return self._vector_mul(other)
        if isinstance(other, BlockLowRank):
            return BlockLowRank(numpy.matmul(self._b, other._b))
        return NotImplemented

    def __rmul__(self, other):
        """Return other * self."""
        if isinstance(other, numbers.Number):
            return other * self._b
        if isinstance(other, core.Vector):
            return self._vector_rmul(other)
        return NotImplemented

    def _vector_mul(self, other):
        """Return self * other (Vector object)."""
        return NotImplemented

    def _vector_rmul(self, other):
        """Return other (Vector object) * self."""
        return NotImplemented

    def __pow__(self, other):
        """Return self ** other."""
        return BlockLowRank(numpy.linalg.matrix_power(self._b, other))

    def __rpow__(self, other):
        """Return other ** self."""
        return BlockLowRank(numpy.linalg.matrix_power(self._b, other))

    def __eq__(self, other):
        """Return self == other."""
        if isinstance(other, BlockLowRank):
            return (self._b == other._b).all()
        return False

    def __ne__(self, other):
        """Return self != other."""
        return not (self == other)

    def __array__(self):
        """Return a reference to self."""
        return self._b

    def to_dense(self):
        """Return self as Dense object."""
        return Dense(numpy.block(self._b.tolist()))


def build(obj, nb, dense_idx=None, method="svd", eps=None, rank=None):
    """Return block low rank (BLR) Matrix.

    Parameters
    ----------
    obj: array_like
        2 dimensional array object.
    nb: int
        A number of blocks.
    dense_idx: list of tuple, default None 
        This is a list of tuple which specifies which block index should
        be Dense object. If you do not give this parameter, only the block
        diagonals are to be Dense object.
    method: str, default 'svd' 
        A approximation method name. You can choose it from following
        list.

        1. 'svd'
            Singular Value Decomposition Method.
        2. 'aca'
            Adaptive Cross Approximation.
    eps: float, default None
        Numerical value for controlled accuracy.
    rank: int, default None
        Numerical rank for fixed rank approximation.

    Returns
    -------
        BlockLowRank :
            BLR matrix satisfying the conditions you gave.

    Examples
    --------
    You can generate the BLR matrix which has LR matrices
    at non-diagonals as following.

    >>> from scipy.linalg import hilbert
    >>> import blrlib as bl
    >>> X = bl.build_blrmatrix(hilbert(1000), nb=4, method="svd", eps=1e-4)
    >>> print(X)
    BlockLowRank(4x4)
    (0, 0): Dense(250x250)
    (0, 1): LowRank(250x250, 3)
    (0, 2): LowRank(250x250, 2)
    ...
    (3, 3): Dense(250x250)
    """
    if not isinstance(obj, numpy.ndarray):
        obj = numpy.array(obj)
    if obj.ndim != 2:
        raise ValueError("'obj' must be 2 dimensional")
    if not isinstance(nb, int) or nb < 1:
        raise ValueError("'nb' must be positive integer")
    if not dense_idx:
        dense_idx = [(i, i) for i in range(nb)]

    B = numpy.full((nb, nb), None)
    m, n = obj.shape

    rslices = [m // nb for _ in range(nb - 1)]
    rslices.append(m // nb + m % nb)
    rslices = numpy.array(rslices, dtype=numpy.int)
    cslices = [n // nb for _ in range(nb - 1)]
    cslices.append(n // nb + n % nb)
    cslices = numpy.array(cslices, dtype=numpy.int)

    for i, j in numpy.ndindex(B.shape):
        rstart, rend = rslices[:i].sum(), rslices[:i + 1].sum()
        cstart, cend = cslices[:j].sum(), cslices[:j + 1].sum()

        if (i, j) in dense_idx:
            B[i, j] = Dense(obj[rstart:rend, cstart:cend])
        else:
            B[i, j] = LowRank(obj[rstart:rend, cstart:cend], method, eps, rank)

    return BlockLowRank(B)
