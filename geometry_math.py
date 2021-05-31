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
from . import ngon_mesh_refine, surface_vector, triangle_mesh_laplacian


DEBUG_METHOD = False


def compute_heat(bm, source_reader, obstacle_reader, heat_writer, time_scale=1.0):
    trimesh, P = ngon_mesh_refine.triangulate_mesh(bm)

    # Optional obstacle factors
    obstacle = obstacle_reader.read_scalar(bm) if obstacle_reader else np.zeros(len(bm.verts))
    vertex_stiffness = P @ (1.0 - obstacle)
    # vertex_stiffness = np.ones(trimesh.verts.shape[0])
    # print(np.array2string(vertex_stiffness, max_line_width=500, threshold=50000))

    # Build Laplacian
    Af, Mf, Sf, Gf = triangle_mesh_laplacian.compute_laplacian(trimesh, vertex_stiffness)

    log_matrix(Af, "Af")
    log_matrix(Mf, "Mf")
    log_matrix(Sf, "Sf")
    log_matrix(Gf, "Gf")
    # print(np.array2string(Gf.todense(), max_line_width=500, threshold=50000))

    # Combine into coarse mesh matrices
    # Diagonalize mass matrix by lumping rows together
    M = P.transpose() @ Mf @ P
    M = sparse.dia_matrix((M.sum(axis=1).transpose(), 0), M.shape)
    S = P.transpose() @ Sf @ P
    log_matrix(M, "M")
    log_matrix(S, "S")

    # TODO this could be better
    Af = sparse.diags(np.repeat(Af, 3, axis=0), 0)
    Df = -Gf.transpose() * Af
    log_matrix(Df, "Df")
    G = Gf @ P
    log_matrix(G, "G")
    D = P.transpose() @ Df
    log_matrix(D, "D")

    # Mean square edge length is used as a "time step" in heat flow solving.
    t = time_scale * sum(((edge.verts[0].co - edge.verts[1].co).length_squared for edge in bm.edges), 0.0) / len(bm.edges) if bm.edges else 0.0

    # Solve the heat equation: (M - t*S) * u = b
    source = source_reader.read_scalar(bm)
    u = sparse.linalg.spsolve(M - t*S, source)
    log_matrix(u, "u")

    if heat_writer:
        heat_writer.write_scalar(bm, u)

    return M, S, G, D, u


def compute_distance(bm, source_reader, obstacle_reader, distance_writer):
    M, S, G, D, u = compute_heat(bm, source_reader, obstacle_reader, None)

    # Normalized gradient
    g = G @ u
    g = np.reshape(g, (-1, 3))
    gnorm = np.expand_dims(np.linalg.norm(g, axis=1), axis=1)
    g = -np.divide(g, gnorm, out=np.zeros_like(g), where=gnorm!=0)
    log_matrix(g, "g")

    d = sparse.linalg.spsolve(S, D @ g.flatten())
    d -= np.amin(d)
    log_matrix(d, "d")

    if distance_writer:
        distance_writer.write_scalar(bm, d)

    return M, S, G, D, u, d


def compute_parallel_transport(bm, source_reader, obstacle_reader, vector_writer):
    totverts = len(bm.verts)

    source = source_reader.read_vector(bm)

    vsurf, vnormal = surface_vector.project_vectors_on_surface(bm, source)
    # vsurf = np.full(totverts, 0.255 + 1.63j, dtype=complex)

    world_vectors = surface_vector.surface_vectors_to_world(bm, vsurf, np.zeros_like(vnormal))

    vector_writer.write_vector(bm, world_vectors)

    # M, S, G, D, u = compute_heat(bm, source_reader, obstacle_reader, None)

    # # Normalized gradient
    # g = G @ u
    # g = np.reshape(g, (-1, 3))
    # gnorm = np.expand_dims(np.linalg.norm(g, axis=1), axis=1)
    # g = -np.divide(g, gnorm, out=np.zeros_like(g), where=gnorm!=0)
    # log_matrix(g, "g")

    # d = sparse.linalg.spsolve(S, D @ g.flatten())
    # d -= np.amin(d)
    # log_matrix(d, "d")

    # if distance_writer:
    #     distance_writer.write_scalar(bm, d)

    # return M, S, G, D, u, d
