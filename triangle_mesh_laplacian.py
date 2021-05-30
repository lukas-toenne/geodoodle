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

import bpy
from math import sqrt, cos, sin, pi, atan2
from mathutils import Vector
from .util import *
import numpy as np
from scipy import sparse
from scipy.sparse import linalg



# Computes a Laplacian for a triangle mesh.
# The Laplace operator matrix is returned as a composition of a "mass" matrix M and a "stiffness" matrix S:
#   L = M^-1 @ S
#
# In addition the gradient operator G and the face areas A are also computed.
# The divergence operator D can be computed from this as:
#   D = -G^T @ A
#
# Parameters
# ----------
# verts : ndarray of shape (V, 3) and type float
#   Array of vertex positions.
# triangles : ndarray of shape (T, 3) and type int
#   Index triplets describing triangles of the mesh
# vertex_stiffness : ndarray of size V and type float
#   Stiffness factor per vertex.
#
# Returns
# -------
# A : ndarray of size T and type float
#   Triangle area list, can be used to compute divergence from the gradient.
# M : spmatrix of shape (V, V) and type float
#   Mass matrix representing inertia in heat propagation.
# S : spmatrix of shape (V, V) and type float
#   Stiffness matrix representing heat propagation speed along edges.
# G : spmatrix of shape (V * 3, V) and type float
#   Gradient matrix for computing the gradient of a scalar field on vertices.
def compute_laplacian(verts : np.ndarray, triangles : np.ndarray, vertex_stiffness : np.ndarray):
    numverts = verts.shape[0]
    numtris = triangles.shape[0]

    idx_a = triangles[:,0]
    idx_b = triangles[:,1]
    idx_c = triangles[:,2]

    vert_a = verts[idx_a,:]
    vert_b = verts[idx_b,:]
    vert_c = verts[idx_c,:]

    edge_ab = vert_b - vert_a
    edge_bc = vert_c - vert_b
    edge_ca = vert_a - vert_c

    normal, norlen = vector_normalized(-np.cross(edge_ab, edge_ca))

    A = norlen / 2

    # Mass contributes to each triangle corner
    mass = A / 12 # 1/12 contribution to each corner
    M_rows = np.r_[idx_a,  idx_a,  idx_a,  idx_b,  idx_b,  idx_b,  idx_c,  idx_c,  idx_c]
    M_cols = np.r_[idx_a,  idx_b,  idx_c,  idx_a,  idx_b,  idx_c,  idx_a,  idx_b,  idx_c]
    M_data = np.r_[2*mass, mass,   mass,   mass,   2*mass, mass,   mass,   mass,   2*mass]
    M = sparse.coo_matrix((M_data, (M_rows, M_cols)), shape=(numverts, numverts))

    # Cotan stiffness matrix for triangle mesh
    dot_ab = np.sum(edge_bc * edge_ca, axis=1)
    dot_bc = np.sum(edge_ca * edge_ab, axis=1)
    dot_ca = np.sum(edge_ab * edge_bc, axis=1)
    cross_ab = vector_lengths(np.cross(edge_ca, edge_bc))
    cross_bc = norlen # already computed: vector_lengths(np.cross(edge_ab, edge_ca))
    cross_ca = vector_lengths(np.cross(edge_bc, edge_ab))
    cot_ab = vector_safe_divide(dot_ab, cross_ab)
    cot_bc = vector_safe_divide(dot_bc, cross_bc)
    cot_ca = vector_safe_divide(dot_ca, cross_ca)
    stiff_a = vertex_stiffness[idx_a]
    stiff_b = vertex_stiffness[idx_b]
    stiff_c = vertex_stiffness[idx_c]
    stiff_ab = -0.5 * cot_ab * stiff_a * stiff_b
    stiff_bc = -0.5 * cot_bc * stiff_b * stiff_c
    stiff_ca = -0.5 * cot_ca * stiff_c * stiff_a
    S_rows = np.r_[idx_a,  idx_a,  idx_a,  idx_b,  idx_b,  idx_b,  idx_c,  idx_c,  idx_c]
    S_cols = np.r_[idx_a,  idx_b,  idx_c,  idx_a,  idx_b,  idx_c,  idx_a,  idx_b,  idx_c]
    S_data = np.r_[-stiff_ab - stiff_ca, stiff_ab, stiff_ca, stiff_ab, -stiff_bc - stiff_ab, stiff_bc, stiff_ca, stiff_bc, -stiff_ca - stiff_bc]
    S = sparse.coo_matrix((S_data, (S_rows, S_cols)), shape=(numverts, numverts))

    edgenor_a = -np.cross(normal, edge_bc)
    edgenor_b = -np.cross(normal, edge_ca)
    edgenor_c = -np.cross(normal, edge_ab)
    gradient_a = 0.5 * vector_safe_divide(edgenor_a, A[:,None])
    gradient_b = 0.5 * vector_safe_divide(edgenor_b, A[:,None])
    gradient_c = 0.5 * vector_safe_divide(edgenor_c, A[:,None])

    # Order of entries in G:
    #
    # G[0]       = G(triangle 0, point A).x
    # G[1]       = G(triangle 0, point A).y
    # G[2]       = G(triangle 0, point A).z
    # G[3]       = G(triangle 0, point B).x
    # G[4]       = G(triangle 0, point B).y
    # G[5]       = G(triangle 0, point B).z
    # G[6]       = G(triangle 0, point C).x
    # G[7]       = G(triangle 0, point C).y
    # G[8]       = G(triangle 0, point C).z

    # G[9]       = G(triangle 1, point A).x
    # G[10]      = G(triangle 1, point A).y
    # ...
    # G[T*9 - 2] = G(triangle T, point C).y
    # G[T*9 - 1] = G(triangle T, point C).z

    G_rows = (np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])[None,:] + np.arange(0, 3*numtris, 3)[:,None]).flatten()
    G_cols = np.repeat(triangles, 3)
    G_data = np.hstack((gradient_a, gradient_b, gradient_c)).flatten()
    G = sparse.coo_matrix((G_data, (G_rows, G_cols)), shape=(numtris * 3, numverts))

    return A, M, S, G
