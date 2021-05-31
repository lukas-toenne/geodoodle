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
from .triangle_mesh import TriangleMesh
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from dataclasses import dataclass


# Triangulate a mesh by generating internal triangles for ngons.
#
# Based on:
# Bunge, Astrid, et al. "Polygon laplacian made simple." Computer Graphics Forum. Vol. 39. No. 2. 2020.
#
# This method is designed to produce a virtual ngon center that is most suitable
# for Laplacians on the resulting triangle mesh.
# The triangulated mesh is only used as an intermediate structure and not a final output.
# The elongation matrix P is used to map from the original vertex set
# to the refined mesh vertices and vice versa.
#
# Parameters
# ----------
# bm : BMesh
#   Original input mesh that is triangulated.
#
# Returns
# -------
# trimesh : TriangleMesh 
#   Data class that stores vertex positions and triangles of the refined mesh.
# P : ndarray of shape (V + F, V) and type float
#   Elongation matrix that maps original vertices to refined mesh vertices.
def triangulate_mesh(bm):
    bm.verts.index_update()
    bm.faces.index_update()

    totverts = len(bm.verts)
    totfaces = len(bm.faces)
    # TODO any faster way to compute this? seems bm.loops has no len()
    totloops = sum(len(face.verts) for face in bm.faces)

    # Vertex positions
    vert_pos = np.fromiter((c for vert in bm.verts for c in vert.co[:]), dtype=float, count=totverts * 3).reshape(-1, 3)

    # Vertex count per face
    face_vert_count = np.fromiter((len(face.loops) for face in bm.faces), dtype=int, count=totfaces)
    # Loop start and stop index per face
    face_stop_idx = np.cumsum(face_vert_count)
    face_start_idx = np.r_[0, face_stop_idx[:-1]]

    # Vertex index for each loop
    loop_idx = np.fromiter((loop.vert.index for face in bm.faces for loop in face.loops), dtype=int, count=totloops)
    # Vertex index for each loop, shifted left for edge rings
    loop_idx_next = np.fromiter((loop.link_loop_next.vert.index for face in bm.faces for loop in face.loops), dtype=int, count=totloops)
    # Owning face index for each loop
    loop_face_idx = np.fromiter((face.index for face in bm.faces for loop in face.loops), dtype=int, count=totloops)

    # Edge vector at each loop
    loop_edge = vert_pos[loop_idx_next,:] - vert_pos[loop_idx,:]

    # Weight matrix maps vertex positions to face center vertices.
    # Forms the lower part of the elongation matrix P.
    W_builder = CooBuilder(shape=(totfaces, totverts), entries=totloops, dtype=float)

    # Solve minimization problem to determine weights for the virtual center vertex.
    # This computes weights such that the squared area of the fan triangles is minimized.
    # See the paper for details: Bunge et al., Polygon Laplacian Made Simple
    for face in bm.faces:
        start_idx = face_start_idx[face.index]
        stop_idx = face_stop_idx[face.index]
        numverts = face_vert_count[face.index]
        face_verts = vert_pos[loop_idx[start_idx:stop_idx],:]
        face_verts_next = vert_pos[loop_idx_next[start_idx:stop_idx],:]
        edges = face_verts_next - face_verts

        # Per-face weight minimization matrix for computing the virtual center vertex.
        # Matrix Q is a (n, n, 3) matrix of cross products for each face vertex with each edge vector.
        # Matrix A describes the surface energy to be minimized.
        Q = np.cross(face_verts[:,None,:], edges[None,:,:])
        A = np.einsum("ikl,jkl", Q, Q)
        b = np.einsum("ikl,kkl", Q, Q)
        # Add row of ones to enforce unity
        A = np.vstack((A, np.ones(numverts)))
        b = np.hstack((b, np.ones(1)))

        # Solve weights
        weights, res, rank, sing = np.linalg.lstsq(A, b, rcond=None)

        # Add entries in the weight matrix: row at face.index contains weights for the face center vertex.
        W_builder.extend(np.full(numverts, face.index), loop_idx[start_idx:stop_idx], weights)

    # Computes face centers from vertex positions, shape is (totfaces, totverts)
    center_weights = W_builder.construct()

    # Elongation matrix P: Maps vertices of the original mesh to vertices of the refined (subdivided) mesh.
    #   First totverts rows are identity matrix (old verts are part of the refined mesh).
    #   Lower totfaces rows are weights for constructing a virtual vertex for each face
    P = sparse.vstack((sparse.identity(totverts), center_weights))

    # Map original vertex positions to new vertices
    new_vert_pos = P @ vert_pos

    # Form new triangles:
    # - 1st index is the new face center vertex, starting after original vertices
    # - 2nd and 3rd index are the original loop vertices
    new_triangles = np.column_stack((loop_face_idx + totverts, loop_idx, loop_idx_next))

    trimesh = TriangleMesh(new_vert_pos, new_triangles)
    return trimesh, P
