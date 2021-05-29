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
from dataclasses import dataclass
from . import surface_vector


# UNUSED
# Per-face Laplacian as defined by
# ALEXA M., WARDETZKY M.: Discrete Laplacians on general polygonal meshes
def get_face_laplacian_AW11(face):
    def get_face_edge_matrix(face):
        numedges = len(face.loops)
        def values():
            for loop in face.loops:
                E = loop.link_loop_next.vert.co - loop.vert.co
                yield from E
        E = np.fromiter(values(), dtype=float, count=numedges * 3)
        E.shape = numedges, 3
        return E

    def get_face_midpoint_matrix(face):
        numedges = len(face.loops)
        def values():
            for loop in face.loops:
                B = 0.5 * (loop.link_loop_next.vert.co + loop.vert.co)
                yield from B
        B = np.fromiter(values(), dtype=float, count=numedges * 3)
        B.shape = numedges, 3
        return B

    E = get_face_edge_matrix(face)
    B = get_face_midpoint_matrix(face)

    A = np.transpose(E) @ B
    M = B @ np.transpose(B) * sqrt(2.0) / np.linalg.norm(A)

    n = np.array([-A[1,2], A[0, 2], -A[0, 1]])
    nlen = np.sqrt((n**2).sum())
    n = n / nlen if nlen > 0 else np.zeros(3)

    E_proj = E - np.outer(np.dot(E, n), n)

    C = linalg.null_space(np.transpose(E_proj))

    U = planarity_factor * np.identity(C.shape[1])
    Mf = M + C @ U @ np.transpose(C)

    return Mf


# @dataclass
# class GeoOpResults:
#     """Intermediate data generator during geometry operations"""
#     M : sparse.spmatrix
#     S : sparse.spmatrix
#     G : sparse.spmatrix
#     D : sparse.spmatrix
#     u : np.ndarray


# Computes matrices:
#   Mass M
#   Stiffness S
#   Gradient G
#   Divergence D
def compute_laplacian(bm, vertex_stiffness):
    totverts = len(bm.verts)
    totfaces = len(bm.faces)
    # TODO any faster way to compute this? seems bm.loops has no len()
    totloops = sum(len(face.verts) for face in bm.faces)

    refinedverts = totverts + totfaces # One virtual vertex added per face
    refinedfaces = totloops # Each face replaced by a triangle fan

    # Elongation matrix P: Maps vertices of the original mesh to vertices of the refined (subdivided) mesh.
    #   First totverts rows are identity matrix (old verts are part of the refined mesh).
    #   Lower totfaces rows are weights for constructing a virtual vertex for each face
    P_builder = CooBuilder(shape=(refinedverts, totverts), entries=totverts+totloops, dtype=float)
    P_builder.extend(np.arange(totverts, dtype=int), np.arange(totverts, dtype=int), np.ones(totverts, dtype=float))

    # Face areas for computing divergence from gradient operator
    Af = np.zeros((refinedfaces))

    # Mass matrix Mf (for the refined mesh): Triangle areas determine inertia in heat propagation.
    # Multiple entries for vertices are expected, these get summed up by the coo_matrix constructor.
    Mf_builder = CooBuilder(shape=(refinedverts, refinedverts), entries=9*totloops, dtype=float)

    # Stiffness matrix Sf: Cotan of adjacent angles defines heat propagation speed along edges.
    # Multiple entries for vertices are expected, these get summed up by the coo_matrix constructor.
    Sf_builder = CooBuilder(shape=(refinedverts, refinedverts), entries=9*totloops, dtype=float)

    # Gradient operator Gf: computes per-loop gradient vector from a scalar field.
    Gf_builder = CooBuilder(shape=(refinedfaces * 3, refinedverts), entries=9*totloops, dtype=float)


    # Slow method of adding laplacian elements for a face with an explicit loop
    def add_face_laplacian_loopy(bm, face, center, weights, loop_count):
        totverts = len(bm.verts)

        for k, loop in enumerate(face.loops):
            loop_idx = loop_count + k
            P_builder.append(totverts + face.index, loop.vert.index, weights[k])

            idx_a = loop.vert.index
            idx_b = loop.link_loop_next.vert.index
            idx_c = totverts + face.index
            ab = loop.link_loop_next.vert.co - loop.vert.co
            bc = center - loop.link_loop_next.vert.co
            ca = loop.vert.co - center

            normal = ca.cross(bc)
            norlen = normal.length
            if norlen > 0:
                normal /= norlen
                area = norlen / 2

                # Area of the fan triangles contributes to face verts and the virtual center vert
                mass = area / 12 # 1/12 contribution to each corner
                Mf_builder.append(idx_a, idx_a, 2.0 * mass)
                Mf_builder.append(idx_a, idx_b, mass)
                Mf_builder.append(idx_a, idx_c, mass)
                Mf_builder.append(idx_b, idx_a, mass)
                Mf_builder.append(idx_b, idx_b, 2.0 * mass)
                Mf_builder.append(idx_b, idx_c, mass)
                Mf_builder.append(idx_c, idx_a, mass)
                Mf_builder.append(idx_c, idx_b, mass)
                Mf_builder.append(idx_c, idx_c, 2.0 * mass)

                # Cotan stiffness matrix for triangle mesh
                stiff_a = vertex_stiffness[idx_a]
                stiff_b = vertex_stiffness[idx_b]
                stiff_ab = -0.25 * bc.dot(ca) / area * stiff_a * stiff_b
                stiff_bc = -0.25 * ca.dot(ab) / area * stiff_b * stiff_c
                stiff_ca = -0.25 * ab.dot(bc) / area * stiff_c * stiff_a
                Sf_builder.append(idx_a, idx_a, -stiff_ab - stiff_ca)
                Sf_builder.append(idx_a, idx_b, stiff_ab)
                Sf_builder.append(idx_a, idx_c, stiff_ca)
                Sf_builder.append(idx_b, idx_a, stiff_ab)
                Sf_builder.append(idx_b, idx_b, -stiff_bc - stiff_ab)
                Sf_builder.append(idx_b, idx_c, stiff_bc)
                Sf_builder.append(idx_c, idx_a, stiff_ca)
                Sf_builder.append(idx_c, idx_b, stiff_bc)
                Sf_builder.append(idx_c, idx_c, -stiff_ca - stiff_bc)

                Af[loop_count + k] = area
                idx_g = (loop_count + k) * 3
                Gf_builder.extend([idx_g+0, idx_g+1, idx_g+2], [idx_a, idx_a, idx_a], (0.5 * normal.cross(bc) / area)[:])
                Gf_builder.extend([idx_g+0, idx_g+1, idx_g+2], [idx_b, idx_b, idx_b], (0.5 * normal.cross(ca) / area)[:])
                Gf_builder.extend([idx_g+0, idx_g+1, idx_g+2], [idx_c, idx_c, idx_c], (0.5 * normal.cross(ab) / area)[:])
            else:
                # Zero entries to fill up coo_matrix construction arrays
                Mf_builder.append(idx_a, idx_a, 0.0)
                Mf_builder.append(idx_a, idx_b, 0.0)
                Mf_builder.append(idx_a, idx_c, 0.0)
                Mf_builder.append(idx_b, idx_a, 0.0)
                Mf_builder.append(idx_b, idx_b, 0.0)
                Mf_builder.append(idx_b, idx_c, 0.0)
                Mf_builder.append(idx_c, idx_a, 0.0)
                Mf_builder.append(idx_c, idx_b, 0.0)
                Mf_builder.append(idx_c, idx_c, 0.0)

                Sf_builder.append(idx_a, idx_a, 0.0)
                Sf_builder.append(idx_a, idx_b, 0.0)
                Sf_builder.append(idx_a, idx_c, 0.0)
                Sf_builder.append(idx_b, idx_a, 0.0)
                Sf_builder.append(idx_b, idx_b, 0.0)
                Sf_builder.append(idx_b, idx_c, 0.0)
                Sf_builder.append(idx_c, idx_a, 0.0)
                Sf_builder.append(idx_c, idx_b, 0.0)
                Sf_builder.append(idx_c, idx_c, 0.0)

                Gf_builder.extend([idx_g+0, idx_g+1, idx_g+2], [idx_a, idx_a, idx_a], (0.0, 0.0, 0.0))
                Gf_builder.extend([idx_g+0, idx_g+1, idx_g+2], [idx_b, idx_b, idx_b], (0.0, 0.0, 0.0))
                Gf_builder.extend([idx_g+0, idx_g+1, idx_g+2], [idx_c, idx_c, idx_c], (0.0, 0.0, 0.0))

    # Per-face Laplacian entries using numpy arrays instead of for loop.
    # Actually slower than the loopy version above, probably only better for extreme ngons.
    # Eventually avoiding looping over faces entirely could amortize this again,
    # this variant of computing face entries is a step in that direction.
    #
    # Cross products on small arrays of 3-vectors have quite a bit of overhead,
    # might try replacing with explicit add/mul, but could also work better for full-mesh arrays.
    def add_face_laplacian_np(bm, face, center, weights, loop_count):
        numverts = len(face.verts)

        idx_a = np.fromiter((loop.vert.index for loop in face.loops), dtype=int, count=numverts)
        idx_b = np.roll(idx_a, -1)
        idx_c = np.full(numverts, face.index + totverts)

        P_builder.extend(idx_c, idx_a, weights)

        loop_verts = np.fromiter((c for loop in face.loops for c in loop.vert.co[:]), dtype=float, count=numverts * 3).reshape(numverts, 3)
        loop_verts_next = np.roll(loop_verts, -1, axis=0)
        ab = loop_verts_next - loop_verts
        bc = center[None,:] - loop_verts_next
        ca = loop_verts - center[None,:]

        normal = -np.cross(ca, bc)
        norlen = np.linalg.norm(normal, axis=1)
        normal = np.divide(normal, norlen[:,None], out=np.zeros_like(normal), where=norlen[:,None] != 0.0)
        area = norlen / 2

        # Area of the fan triangles contributes to face verts and the virtual center vert
        mass = area / 12 # 1/12 contribution to each corner
        Mf_builder.extend_np(idx_a, idx_a, 2.0 * mass)
        Mf_builder.extend_np(idx_a, idx_b, mass)
        Mf_builder.extend_np(idx_a, idx_c, mass)
        Mf_builder.extend_np(idx_b, idx_a, mass)
        Mf_builder.extend_np(idx_b, idx_b, 2.0 * mass)
        Mf_builder.extend_np(idx_b, idx_c, mass)
        Mf_builder.extend_np(idx_c, idx_a, mass)
        Mf_builder.extend_np(idx_c, idx_b, mass)
        Mf_builder.extend_np(idx_c, idx_c, 2.0 * mass)

        # Cotan stiffness matrix for triangle mesh
        stiff_a = vertex_stiffness[idx_a]
        stiff_b = vertex_stiffness[idx_b]
        cos_ab = np.sum(bc * ca, axis=1)
        cos_bc = np.sum(ca * ab, axis=1)
        cos_ca = np.sum(ab * bc, axis=1)
        cot_ab = np.divide(cos_ab, area, out=np.zeros_like(cos_ab), where=area != 0.0)
        cot_bc = np.divide(cos_bc, area, out=np.zeros_like(cos_bc), where=area != 0.0)
        cot_ca = np.divide(cos_ca, area, out=np.zeros_like(cos_ca), where=area != 0.0)
        stiff_ab = -0.25 * cot_ab * stiff_a * stiff_b
        stiff_bc = -0.25 * cot_bc * stiff_b * stiff_c
        stiff_ca = -0.25 * cot_ca * stiff_c * stiff_a
        Sf_builder.extend_np(idx_a, idx_a, -stiff_ab - stiff_ca)
        Sf_builder.extend_np(idx_a, idx_b, stiff_ab)
        Sf_builder.extend_np(idx_a, idx_c, stiff_ca)
        Sf_builder.extend_np(idx_b, idx_a, stiff_ab)
        Sf_builder.extend_np(idx_b, idx_b, -stiff_bc - stiff_ab)
        Sf_builder.extend_np(idx_b, idx_c, stiff_bc)
        Sf_builder.extend_np(idx_c, idx_a, stiff_ca)
        Sf_builder.extend_np(idx_c, idx_b, stiff_bc)
        Sf_builder.extend_np(idx_c, idx_c, -stiff_ca - stiff_bc)

        Af[loop_count:loop_count + numverts] = area
        idx_g = np.arange(loop_count * 3, (loop_count + numverts) * 3)
        edgenor_a = -np.cross(normal, bc)
        edgenor_b = -np.cross(normal, ca)
        edgenor_c = -np.cross(normal, ab)
        gradient_a = 0.5 * np.divide(edgenor_a, area[:,None], out=np.zeros_like(edgenor_a), where=area[:,None] != 0.0)
        gradient_b = 0.5 * np.divide(edgenor_b, area[:,None], out=np.zeros_like(edgenor_b), where=area[:,None] != 0.0)
        gradient_c = 0.5 * np.divide(edgenor_c, area[:,None], out=np.zeros_like(edgenor_c), where=area[:,None] != 0.0)
        Gf_builder.extend_np(idx_g, np.repeat(idx_a, 3), gradient_a.flatten())
        Gf_builder.extend_np(idx_g, np.repeat(idx_b, 3), gradient_b.flatten())
        Gf_builder.extend_np(idx_g, np.repeat(idx_c, 3), gradient_c.flatten())

    loop_count = 0
    for face in bm.faces:
        numverts = len(face.verts)
        # TODO virtual vertex only needed for polygons, handle triangles with simple cotan

        def X_values():
            for loop in face.loops:
                yield from loop.vert.co
        X = np.fromiter(X_values(), dtype=float, count=numverts * 3).reshape(numverts, 3)

        # Solve minimization problem to determine weights for the virtual vertex.
        # This computes weights such that the squared area of the fan triangles is minimized.
        # See the paper for details: Bunge et al., Polygon Laplacian Made Simple

        # Cached cross products
        def C_values():
            for iloop in face.loops:
                for jloop in face.loops:
                    yield from iloop.vert.co.cross(jloop.vert.co)
        C = np.fromiter(C_values(), dtype=float, count=numverts * numverts * 3).reshape(numverts, numverts, 3)
        Cnext = np.roll(C, -1, axis=1)

        def A_values():
            for i in range(numverts):
                for j in range(numverts):
                    yield 2.0 * sum((np.dot(Cnext[j, k, :] - C[j, k, :], Cnext[i, k, :] - C[i, k, :]) for k in range(numverts)), 0.0)
        A = np.fromiter(A_values(), dtype=float, count=numverts * numverts).reshape(numverts, numverts)
        # Add row of ones to enforce unity
        A = np.vstack((A, np.ones(numverts)))

        def b_values():
            for i, iloop in enumerate(face.loops):
                yield 2.0 * sum((np.dot(Cnext[i, k, :] - C[i, k, :], Cnext[k, k, :]) for k in range(numverts)), 0.0)
        b = np.fromiter(b_values(), dtype=float, count=numverts)
        # Add row of ones to enforce unity
        b = np.hstack((b, np.ones(1)))

        # Solve weights
        w, res, rank, sing = np.linalg.lstsq(A, b, rcond=None)

        # Virtual vertex
        center = w @ X
        stiff_c = sum((w[k] * vertex_stiffness[loop.vert.index] for k, loop in enumerate(face.loops)), 0.0)

        add_face_laplacian_loopy(bm, face, Vector(center), w, loop_count)
        # add_face_laplacian_np(bm, face, center, w, loop_count)

        loop_count += len(face.verts)
    P = P_builder.construct()
    Mf = Mf_builder.construct()
    Sf = Sf_builder.construct()
    Gf = Gf_builder.construct()
    log_matrix(P, "P")
    log_matrix(Mf, "Mf")
    log_matrix(Sf, "Sf")
    log_matrix(Gf, "Gf")

    # Combine into coarse mesh matrices
    M = P.transpose() @ Mf @ P
    S = P.transpose() @ Sf @ P
    log_matrix(S, "S")

    # Diagonalize mass matrix by lumping rows together
    M = sparse.dia_matrix((M.sum(axis=1).transpose(), 0), M.shape)
    log_matrix(M, "M")

    Af = sparse.diags(np.repeat(Af, 3, axis=0), 0)
    log_matrix(Af, "Af")
    Df = -Gf.transpose() * Af
    log_matrix(Df, "Df")
    G = Gf @ P
    log_matrix(G, "G")
    D = P.transpose() @ Df
    log_matrix(D, "D")

    return M, S, G, D


def compute_heat(bm, source_reader, obstacle_reader, heat_writer, time_scale=1.0):
    numverts = len(bm.verts)
    bm.verts.index_update()
    bm.faces.index_update()

    # Build Laplacian
    vertex_stiffness = 1.0 - obstacle_reader.read_scalar(bm) if obstacle_reader else np.ones(numverts)
    M, S, G, D = compute_laplacian(bm, vertex_stiffness)

    # Mean square edge length is used as a "time step" in heat flow solving.
    t = time_scale * sum(((edge.verts[0].co - edge.verts[1].co).length_squared for edge in bm.edges), 0.0) / len(bm.edges) if bm.edges else 0.0

    # Solve the heat equation: (M - t*S) * u = b
    source = source_reader.read_scalar(bm)
    u = sparse.linalg.spsolve(M - t*S, source)
    log_matrix(u, "u")
    # print(np.array2string(u, max_line_width=500))

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
