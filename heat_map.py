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

from math import sqrt
from mathutils import Vector
import numpy as np
from scipy import linalg

class HeatMapGenerator:
    def __init__(self, bm):
        self.bm = bm

    def vgroup_to_array(self, vgroup, default_weight):
        def values():
            dvert_lay = self.bm.verts.layers.deform.verify()
            idx = vgroup.index
            for vert in self.bm.verts:
                dvert = vert[dvert_lay]
                yield dvert[idx] if idx in dvert else default_weight
        return np.fromiter(values(), dtype=float, count=len(self.bm.verts))

    def vgroup_from_array(self, vgroup, array):
        assert(array.size == len(self.bm.verts))
        dvert_lay = self.bm.verts.layers.deform.verify()
        idx = vgroup.index
        for value, vert in zip(np.nditer(array), self.bm.verts):
            dvert = vert[dvert_lay]
            dvert[idx] = value

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

    # Computes mass and stiffness matrices for the mesh
    def compute_laplacian(self):
        totverts = len(self.bm.verts)
        totfaces = len(self.bm.faces)
        self.bm.verts.index_update()
        self.bm.faces.index_update()

        refinedverts = totverts + totfaces # One virtual vertex added per face
        P = np.vstack((np.identity(totverts), np.zeros((totfaces, totverts))))
        Sf = np.zeros((refinedverts, refinedverts))
        Mf = np.zeros((refinedverts, refinedverts))

        # debug_verts = []
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
            # debug_verts.append(Vector(v))

            # TODO optimize me
            for k, loop in enumerate(face.loops):
                P[totverts + face.index, loop.vert.index] = w[k]

                idx_a = loop.vert.index
                idx_b = loop.link_loop_next.vert.index
                idx_c = totverts + face.index
                ab = loop.link_loop_next.vert.co - loop.vert.co
                bc = center - loop.link_loop_next.vert.co
                ca = loop.vert.co - center

                tricross = ca.cross(bc).length
                if tricross > 0:
                    # Area of the fan triangles contributes to face verts and the virtual center vert
                    mass = tricross / 24 # Area is 0.5 of cross product, times 1/12 contribution to each corner
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
                    stiff_ab = 0.5 * abs(bc.dot(ca)) / tricross
                    stiff_bc = 0.5 * abs(ca.dot(ab)) / tricross
                    stiff_ca = 0.5 * abs(ab.dot(bc)) / tricross
                    Sf[idx_a, idx_b] += stiff_ab
                    Sf[idx_b, idx_a] += stiff_ab
                    Sf[idx_b, idx_c] += stiff_bc
                    Sf[idx_c, idx_b] += stiff_bc
                    Sf[idx_c, idx_a] += stiff_ca
                    Sf[idx_a, idx_c] += stiff_ca
                    Sf[idx_a, idx_a] -= stiff_ab + stiff_ca
                    Sf[idx_b, idx_b] -= stiff_bc + stiff_ab
                    Sf[idx_c, idx_c] -= stiff_ca + stiff_bc

        # Combine into coarse mesh matrices
        M = np.transpose(P) @ (Mf @ P)
        S = np.transpose(P) @ (Sf @ P)
        # print("==== M ====")
        # print(np.array2string(M, max_line_width=500))
        # print("==== S ====")
        # print(np.array2string(S, max_line_width=500))

        # Diagonalize mass matrix by lumping rows together
        M = np.diag(np.sum(M, axis=1))

        return M, S

    def debug_build_virtual_verts(self, virtual_verts):
        assert(len(virtual_verts) == len(self.bm.faces))
        old_faces = list(self.bm.faces)
        for oface, vv in zip(old_faces, virtual_verts):
            nvert = self.bm.verts.new(vv)
            for loop in oface.loops:
                self.bm.faces.new((nvert, loop.vert, loop.link_loop_next.vert))
            self.bm.faces.remove(oface)

    def generate(self, boundary_vg, output_vg, time_scale=1.0):
        # Build Laplacian
        M, S = self.compute_laplacian()

        # Mean square edge length is used as a "time step" in heat flow solving.
        t = time_scale * sum(((edge.verts[0].co - edge.verts[1].co).length_squared for edge in self.bm.edges), start=0.0) / len(self.bm.edges) if self.bm.edges else 0.0

        # Solve the heat equation: (M - t*S) * u = b
        boundary = self.vgroup_to_array(boundary_vg, 0.0)
        # print("==== b ====")
        # print(np.array2string(boundary, max_line_width=500))
        u = np.linalg.solve(M - t*S, boundary)
        # print("==== u ====")
        # print(np.array2string(u, max_line_width=500))

        self.vgroup_from_array(output_vg, u)

        # self.debug_build_virtual_verts(debug_verts)
        # print(P)

