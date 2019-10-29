# -*- coding: utf-8 -*-
import numbers
import numpy
from .. import core


class vector(object):
    """A vector object utilizing a numpy.ndarray object.

    Arguments
    ---------
    obj : array like

    Attributes
    ----------
    a : numpy.ndarray
        Self as numpy.ndarray object.
    T : vector
        Transpose of self.
    shape : tuple(int, int)
        Shape of self.
    size : int
        Number of elements in self.
    nbytes : int
        total bytes consumed by the elements of self.
    norm : float
        Euclidean norm of self.

    Examples
    --------
    Ex.1 : Make the vector instance.
    """

    def __init__(self, obj):
        """Initialize self."""
        self._content = numpy.array(obj)

        if self._content.ndim == 0:
            self._content = self._content.reshape(1, 1)
        if self._content.ndim == 1:
            self._content = self._content.reshape(self._content.size, 1)
        if self._content.ndim == 2 and 1 not in self._content.shape:
            raise ValueError("invalid array shape")
        if self._content.ndim > 2:
            raise TypeError("invalid array dimension")
        if self._content.dtype not in (numpy.int, numpy.float, numpy.complex):
            raise TypeError("valid type instances must be set")

    @property
    def a(self):
        """Return self as numpy.ndarray object."""
        return self._content

    @property
    def T(self):
        """Return transpose of the vector."""
        return vector(self._content.T)

    @property
    def shape(self):
        """Return shape of the vector."""
        return self._content.shape

    @property
    def size(self):
        """Return number of elements in the vector."""
        return self._content.size

    @property
    def nbytes(self):
        """Return total bytes consumed by the elements of the vector."""
        return self._content.nbytes

    @property
    def norm(self):
        """Return euclidean norm of the vector."""
        return numpy.linalg.norm(self._content)

    def __repr__(self):
        """Return the official string representation of this object."""
        return "vector\n" + str(self.a)

    def __str__(self):
        """Return the informal string representation of this object."""
        return "vector({0})".format(self.size)

    def __getitem__(self, key):
        """Return self[key]."""
        if not isinstance(key, (slice, int)):
            raise TypeError("invalid key type")

        if self.shape[0] == 1:
            if isinstance(key, int):
                return self.a[0, key]
            return vector(self.a[0, key])
        if isinstance(key, int):
            return self.a[key, 0]
        return vector(self.a[key, 0])

    def __setitem__(self, key, item):
        """Set item to self[key]."""
        if not isinstance(key, (slice, int)):
            raise TypeError("invalid key type")

        if self.shape[0] == 1:
            self.a[0, key] = numpy.array(item)
        self.a[key, 0] = numpy.array(item)

    def __neg__(self):
        """Return -self"""
        return vector(-self.a)

    def __pos__(self):
        """Return +self"""
        return self

    def __add__(self, other):
        """Return self + other."""
        if isinstance(other, vector):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return vector(self.a + other.a)
        return NotImplemented

    def __sub__(self, other):
        """Return self - other."""
        if isinstance(other, vector):
            if self.shape != other.shape:
                raise ValueError("shape must be aligned")
            return vector(self.a + (-other.a))
        return NotImplemented

    def __mul__(self, other):
        """Return self * other."""
        if isinstance(other, numbers.Number):
            return vector(self.a * other)
        if isinstance(other, vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape must be aligned")
            if self.shape[1] == 1:
                return core.matrix(self.a @ other.a)
            return (self.a @ other.a).item()
        return NotImplemented

    def __rmul__(self, other):
        """Return other * self."""
        if isinstance(other, numbers.Number):
            return vector(other * self.a)
        return NotImplemented

    def __eq__(self, other):
        """Return self == other."""
        if isinstance(other, vector):
            return numpy.allclose(self.a, other.a)
        return False

    def __ne__(self, other):
        """Return self != other."""
        return not (self == other)

    def __array__(self):
        """Return a reference to self."""
        return self.a
