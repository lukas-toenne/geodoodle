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
from math import sqrt
from mathutils import Vector
import numpy as np


class HeatMapGenerator:
    def __init__(self, bm):
        self.bm = bm

    def get_face_edge_matrix(self, face):
        numedges = len(face.loops)
        def values():
            for loop in face.loops:
                E = loop.link_loop_next.vert.co - loop.vert.co
                yield from E
        E = np.fromiter(values(), dtype=float, count=numedges * 3)
        E.shape = numedges, 3
        return E

    def get_face_midpoint_matrix(self, face):
        numedges = len(face.loops)
        def values():
            for loop in face.loops:
                B = 0.5 * (loop.link_loop_next.vert.co + loop.vert.co)
                yield from B
        B = np.fromiter(values(), dtype=float, count=numedges * 3)
        B.shape = numedges, 3
        return B

    # UNUSED
    # Per-face Laplacian as defined by
    # ALEXA M., WARDETZKY M.: Discrete Laplacians on general polygonal meshes
    def get_face_laplacian_AW11(face):
        from scipy import linalg

        E = self.get_face_edge_matrix(face)
        B = self.get_face_midpoint_matrix(face)

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

    # Computes matrices:
    #   Mass M
    #   Stiffness S
    #   Gradient G
    #   Divergence D
    def compute_laplacian(self):
        totverts = len(self.bm.verts)
        totfaces = len(self.bm.faces)
        # TODO any faster way to compute this? seems bm.loops has no len()
        totloops = sum(len(face.verts) for face in self.bm.faces)

        refinedverts = totverts + totfaces # One virtual vertex added per face
        refinedfaces = totloops # Each face replaced by a triangle fan
        P = np.vstack((np.identity(totverts), np.zeros((totfaces, totverts))))
        Af = np.zeros((refinedfaces))
        Mf = np.zeros((refinedverts, refinedverts))
        Sf = np.zeros((refinedverts, refinedverts))
        Gf = np.zeros((refinedfaces * 3, refinedverts))

        loop_count = 0
        for face in self.bm.faces:
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
                        yield 2.0 * sum((np.dot(Cnext[j, k, :] - C[j, k, :], Cnext[i, k, :] - C[i, k, :]) for k in range(numverts)), start=0.0)
            A = np.fromiter(A_values(), dtype=float, count=numverts * numverts).reshape(numverts, numverts)
            # Add row of ones to enforce unity
            A = np.vstack((A, np.ones(numverts)))

            def b_values():
                for i, iloop in enumerate(face.loops):
                    yield 2.0 * sum((np.dot(Cnext[i, k, :] - C[i, k, :], Cnext[k, k, :]) for k in range(numverts)), start=0.0)
            b = np.fromiter(b_values(), dtype=float, count=numverts)
            # Add row of ones to enforce unity
            b = np.hstack((b, np.ones(1)))

            # Solve weights
            w, res, rank, sing = np.linalg.lstsq(A, b, rcond=None)

            # Virtual vertex
            center = Vector(w @ X)

            # TODO optimize me
            for k, loop in enumerate(face.loops):
                P[totverts + face.index, loop.vert.index] = w[k]

                idx_a = loop.vert.index
                idx_b = loop.link_loop_next.vert.index
                idx_c = totverts + face.index
                ab = loop.link_loop_next.vert.co - loop.vert.co
                bc = center - loop.link_loop_next.vert.co
                ca = loop.vert.co - center

                normal = ca.cross(bc)
                # TODO minor optimize: re-use length
                area = normal.length / 2
                normal.normalize()
                if area > 0:
                    # Area of the fan triangles contributes to face verts and the virtual center vert
                    mass = area / 12 # 1/12 contribution to each corner
                    Mf[idx_a, idx_b] += mass
                    Mf[idx_b, idx_a] += mass
                    Mf[idx_b, idx_c] += mass
                    Mf[idx_c, idx_b] += mass
                    Mf[idx_c, idx_a] += mass
                    Mf[idx_a, idx_c] += mass
                    Mf[idx_a, idx_a] += 2.0 * mass
                    Mf[idx_b, idx_b] += 2.0 * mass
                    Mf[idx_c, idx_c] += 2.0 * mass

                    # Cotan stiffness matrix for triangle mesh
                    stiff_ab = 0.25 * abs(bc.dot(ca)) / area
                    stiff_bc = 0.25 * abs(ca.dot(ab)) / area
                    stiff_ca = 0.25 * abs(ab.dot(bc)) / area
                    Sf[idx_a, idx_b] += stiff_ab
                    Sf[idx_b, idx_a] += stiff_ab
                    Sf[idx_b, idx_c] += stiff_bc
                    Sf[idx_c, idx_b] += stiff_bc
                    Sf[idx_c, idx_a] += stiff_ca
                    Sf[idx_a, idx_c] += stiff_ca
                    Sf[idx_a, idx_a] -= stiff_ab + stiff_ca
                    Sf[idx_b, idx_b] -= stiff_bc + stiff_ab
                    Sf[idx_c, idx_c] -= stiff_ca + stiff_bc

                    Af[loop_count + k] = area
                    idx_g = (loop_count + k) * 3
                    Gf[idx_g:idx_g+3, idx_a] = (0.5 * normal.cross(bc) / area)[:]
                    Gf[idx_g:idx_g+3, idx_b] = (0.5 * normal.cross(ca) / area)[:]
                    Gf[idx_g:idx_g+3, idx_c] = (0.5 * normal.cross(ab) / area)[:]

            loop_count += len(face.verts)

        # Combine into coarse mesh matrices
        M = np.transpose(P) @ (Mf @ P)
        S = np.transpose(P) @ (Sf @ P)

        # Diagonalize mass matrix by lumping rows together
        # TODO return just the 1-dim diagonal vector and multiply component-wise
        M = np.diag(np.sum(M, axis=1))

        Af = np.repeat(Af, 3, axis=0)
        Df = -np.transpose(Gf) * Af
        G = Gf @ P
        D = np.transpose(P) @ Df

        return M, S, G, D

    def debug_build_virtual_verts(self, virtual_verts):
        assert(len(virtual_verts) == len(self.bm.faces))
        old_faces = list(self.bm.faces)
        for oface, vv in zip(old_faces, virtual_verts):
            nvert = self.bm.verts.new(vv)
            for loop in oface.loops:
                self.bm.faces.new((nvert, loop.vert, loop.link_loop_next.vert))
            self.bm.faces.remove(oface)

    def generate(self, boundary_reader, heat_writer, distance_writer, time_scale=1.0):
        self.bm.verts.index_update()
        self.bm.faces.index_update()

        # Build Laplacian
        M, S, G, D = self.compute_laplacian()

        # Mean square edge length is used as a "time step" in heat flow solving.
        t = time_scale * sum(((edge.verts[0].co - edge.verts[1].co).length_squared for edge in self.bm.edges), start=0.0) / len(self.bm.edges) if self.bm.edges else 0.0

        # Solve the heat equation: (M - t*S) * u = b
        boundary = boundary_reader(self.bm)
        u = np.linalg.solve(M - t*S, boundary)
        # print("==== u ====")
        # print(np.array2string(u, max_line_width=500))

        heat_writer(self.bm, u)

        # Normalized gradient
        g = G @ u
        g = np.reshape(g, (-1, 3))
        g = -g / np.expand_dims(np.sqrt(np.sum(np.square(g), axis=1)), axis=1)

        d = np.linalg.solve(S, D @ g.flatten())
        d -= np.amin(d)
        distance_writer(self.bm, d)

