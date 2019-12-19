# -*- coding: utf-8 -*-
import numbers
import numpy
from .. import core


class Vector(object):
    """A Vector object utilizing a ``numpy.ndarray`` object.

    Attributes:
        T (Vector): Transpose of self.
        size (int): Size of self.
        shape (tuple): Shape of self.
        nbytes (int): Total bytes consumed by the elements of self.

    Examples:
        Generate the ``Vector`` instance.

        >>> import blrlib as bl
        >>> a = bl.Vector([1, 2, 3])
        >>> print(a)
        Vector(3)
    """

    def __init__(self, obj):
        """Initialize self.

        Arguments:
            obj (array_like): 1 or 2 dimensional array object.
        """
        self._v = numpy.array(obj)

        if self._v.ndim == 0:
            self._v = self._v.reshape(1, 1)
        if self._v.ndim == 1:
            self._v = self._v.reshape(self._v.size, 1)
        if self._v.ndim == 2 and 1 not in self._v.shape:
            raise ValueError("invalid array shape")
        if self._v.ndim > 2:
            raise TypeError("invalid array dimension")
        if self._v.dtype not in (numpy.int, numpy.float):
            raise TypeError("valid type instances must be set")

    @property
    def T(self):
        """``Vector``: Return transpose of the vector."""
        return Vector(self._v.T)

    @property
    def size(self):
        """``int``: Return size of the vector."""
        return self._v.size

    @property
    def shape(self):
        """``tuple``: Return shape of the vector."""
        return self._v.shape

    @property
    def nbytes(self):
        """``int``: Return total bytes consumed by the elements of
        the vector.
        """
        return self._v.nbytes

    def __repr__(self):
        """Return the official string representation of this object."""
        return "Vector\n" + str(self._v)

    def __str__(self):
        """Return the informal string representation of this object."""
        return "Vector({0})".format(self.size)

    def __getitem__(self, key):
        """Return self[key]."""
        if not isinstance(key, (slice, int)):
            raise TypeError("invalid key type")

        if self.shape[0] == 1:
            if isinstance(key, int):
                return self._v[0, key]
            return Vector(self._v[0, key])
        if isinstance(key, int):
            return self._v[key, 0]
        return Vector(self._v[key, 0])

    def __setitem__(self, key, item):
        """Set item to self[key]."""
        if not isinstance(key, (slice, int)):
            raise TypeError("invalid key type")

        if self.shape[0] == 1:
            self._v[0, key] = numpy.array(item)
        self._v[key, 0] = numpy.array(item)

    def __neg__(self):
        """Return -self"""
        return Vector(-self._v)

    def __pos__(self):
        """Return +self"""
        return self

    def __add__(self, other):
        """Return self + other."""
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return Vector(self._v + other._v)
        return NotImplemented

    def __sub__(self, other):
        """Return self - other."""
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return Vector(self._v + (-other._v))
        return NotImplemented

    def __mul__(self, other):
        """Return self * other."""
        if isinstance(other, numbers.Number):
            return Vector(self._v * other)
        if isinstance(other, Vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            if self.shape[1] == 1:
                return core.Dense(numpy.outer(self._v, other._v))
            return numpy.dot(self._v, other._v).item()
        return NotImplemented

    def __rmul__(self, other):
        """Return other * self."""
        if isinstance(other, numbers.Number):
            return Vector(other * self._v)
        return NotImplemented

    def __eq__(self, other):
        """Return self == other."""
        if isinstance(other, Vector):
            return numpy.allclose(self._v, other._v)
        return False

    def __ne__(self, other):
        """Return self != other."""
        return not (self == other)

    def __array__(self):
        """Return a reference to self."""
        return self._v
