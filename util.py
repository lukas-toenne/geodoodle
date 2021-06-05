# MIT License
#
# Copyright (c) 2021 Lukas Toenne
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy import sparse

# Utility class for building a sparse coordinate matrix.
class CooBuilder:
    def __init__(self, shape, entries, dtype=float):
        self.shape = shape
        self.entries = entries
        self.rows = np.empty(entries, dtype=int)
        self.cols = np.empty(entries, dtype=int)
        self.data = np.empty(entries, dtype=dtype)
        self.index = 0

    def construct(self):
        assert(self.index == self.entries)
        return sparse.coo_matrix((self.data, (self.rows, self.cols)), shape=self.shape)

    def append(self, row, col, data):
        assert(self.index < self.entries)
        self.rows[self.index] = row
        self.cols[self.index] = col
        self.data[self.index] = data
        self.index += 1

    def extend(self, rows, cols, data):
        count = len(data)
        assert(len(rows) == count)
        assert(len(cols) == count)
        assert(self.index < self.entries)
        self.rows[self.index:self.index+count] = rows[:]
        self.cols[self.index:self.index+count] = cols[:]
        self.data[self.index:self.index+count] = data[:]
        self.index += count

    def extend_np(self, rows, cols, data):
        count = data.size
        assert(rows.size == count)
        assert(cols.size == count)
        assert(self.index < self.entries)
        self.rows[self.index:self.index+count] = rows[:]
        self.cols[self.index:self.index+count] = cols[:]
        self.data[self.index:self.index+count] = data[:]
        self.index += count

    def reset(self):
        self.rows = np.empty(self.entries, dtype=int)
        self.cols = np.empty(self.entries, dtype=int)
        self.data = np.empty(self.entries, dtype=self.data.dtype)
        self.index = 0


# Safe division, putting zero when divisor is zero.
#
# Parameters
# ----------
# a : ndarray
#   Nominator array or scalar.
# d
#   Divisor array or scalar.
#
# Returns
# -------
# a_div : ndarray of floats
#   Division result.
def vector_safe_divide(a, d):
    return np.divide(a, d, out=np.zeros_like(a), where=d != 0.0)


# Length of vectors
#
# Parameters
# ----------
# a : ndarray of shape (N, 3)
#   Array of vectors, components in the last dimension.
#
# Returns
# -------
# a_len : ndarray of floats
#   Length of vectors in the input array.
def vector_lengths(a):
    return np.linalg.norm(a, axis=1)


# Normalizes an array of vectors.
#
# Parameters
# ----------
# a : ndarray of shape (N, 3)
#   Array of vectors, components in the last dimension.
#
# Returns
# -------
# a_norm : ndarray like a
#   Array of normalized vectors, components in the last dimension.
# a_len : ndarray of floats
#   Length of vectors in the input array.
def vector_normalized(a):
    a_len = vector_lengths(a)
    a_norm = vector_safe_divide(a, a_len[:,None])
    return a_norm, a_len


# Cumulative sum of rows in a sparse matrix.
# Based on https://www.debugcn.com/en/article/101824839.html
#
# Parameters
# ----------
# a : spmatrix, preferably csr_matrix
#   Sparse matrix whose rows will be summed up.
#
# Returns
# -------
# r : spmatrix like a
#   Sparse matrix where elements are the cumulative sum over rows of a.
def sparse_cumsum(a):
    r = a.copy()
    indptr = r.indptr
    data = r.data
    for i in range(r.shape[0]):
        st = indptr[i]
        en = indptr[i + 1]
        np.cumsum(data[st:en], out=data[st:en])
    return r


def log_matrix(m, name):
    if isinstance(m, np.ndarray):
        size = m.nbytes
        mtype = "numpy"
    else:
        size = m.data.size
        mtype = "scipy"
    if size > 1024.0:
        size /= 1024.0
        if size > 1024.0:
            size /= 1024.0
            unit = "MB"
        else:
            unit = "KB"
    else:
        unit = "Bytes"
    print("{name: <4} {mtype: <8} {shape: <20} {size:.2f} {unit}".format(name=name+":", shape=str(m.shape), mtype="["+mtype+"]", size=size, unit=unit))
    pass


def debug_build_virtual_verts(bm, virtual_verts):
    assert(len(virtual_verts) == len(bm.faces))
    old_faces = list(bm.faces)
    for oface, vv in zip(old_faces, virtual_verts):
        nvert = bm.verts.new(vv)
        for loop in oface.loops:
            bm.faces.new((nvert, loop.vert, loop.link_loop_next.vert))
        bm.faces.remove(oface)
