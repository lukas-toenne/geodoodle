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
from dataclasses import dataclass, field
from .util import *
from math import pi


tau = 2.0 * pi


# Mesh structure with triangles in numpy arrays.
@dataclass
class TriangleMesh:
    # ndarray of shape (V + F, 3) and type float
    # Triangulated vertex positions. First V vectors are original vertices, followed by F vectors at face centers.
    verts : np.ndarray

    # ndarray of shape (T, 3) and type int
    # Index triplets defining triangles of the output mesh.
    # Number of triangles T depends on topology of the input mesh: Triangles remain triangles, while ngons are turned into triangle fans.
    triangles : np.ndarray

    # ndarray of size T and type float
    # Triangle area.
    area : np.ndarray = None

    # ndarray of shape (T, 3) and type float
    # Triangle normals.
    normal : np.ndarray = None

    # ndarray of shape (T, 3) and type float
    # Interior angles for each triangle corner.
    angle : np.ndarray = None

    # ndarray of shape (T, 3) and type float
    # Cotangent of interior angles for each triangle corner.
    cot_angle : np.ndarray = None

    # spmatrix of shape (T * 3, V)
    # Gradient operator which computes a gradient vector per triangle from a scalar field defined on vertices.
    gradient : sparse.spmatrix = None

    # spmatrix of shape (V, T * 3)
    # Divergence operator which computes a scalar field per vertex from a vector field defined on triangles.
    divergence : sparse.spmatrix = None

    # spmatrix of shape (V, V) and type float
    # Mass matrix, representing inertia in heat propagation.
    mass : sparse.spmatrix = None

    # spmatrix of shape (V, V) and type float
    # Stiffness matrix, representing heat propagation speed along edges
    stiffness : sparse.spmatrix = None

    # ndarray of size V and type bool
    # True for vertices that are part of a boundary,
    # i.e. two adjacent edges are part of only one triangle.
    boundary : np.ndarray = None

    # ndarray size sum(loop_counts), i.e. T * 3 or greater if there are boundaries, and type int
    # Vertex indices making up loops around a center vertex.
    # Loop for a vertex i is given by loop_verts_index:
    #   loop_verts[loop_verts_index[i]:loop_verts_index[i+1]]
    loop_verts : np.ndarray = None
    # ndarray of size V + 1 and type int
    # Start index of the loop around each vertex.
    loop_verts_index : np.ndarray = None
    # ndarray of size T * 3 and type int
    # Corner indices making up loops around a center vertex.
    # Loop for a vertex i is given by loop_corner_index:
    #   loop_corners[loop_corners_index[i]:loop_corners_index[i+1]]
    loop_corners : np.ndarray = None
    # ndarray of size V + 1 and type int
    # Start index of the loop around each vertex.
    loop_corners_index : np.ndarray = None

    # ndarray of shape (T, 3) and type float
    # Radial angle minimum for each triangle corner.
    radial_angle_min : np.ndarray = None
    # ndarray of shape (T, 3) and type float
    # Radial angle maximum for each triangle corner.
    radial_angle_max : np.ndarray = None


    # Computes area and normal for each triangle as well as interior angles.
    def measure_triangles(self):
        if self.area is not None:
            return

        idx_a = self.triangles[:,0]
        idx_b = self.triangles[:,1]
        idx_c = self.triangles[:,2]

        vert_a = self.verts[idx_a,:]
        vert_b = self.verts[idx_b,:]
        vert_c = self.verts[idx_c,:]

        edge_ab = vert_b - vert_a
        edge_bc = vert_c - vert_b
        edge_ca = vert_a - vert_c

        # Note dot and cross products here contain edge lengths, cancels out in the cotangent.
        dot_a = np.sum(edge_ca * edge_ab, axis=1)
        dot_b = np.sum(edge_ab * edge_bc, axis=1)
        dot_c = np.sum(edge_bc * edge_ca, axis=1)
        self.normal, cross_a = vector_normalized(-np.cross(edge_ab, edge_ca))
        cross_b = vector_lengths(-np.cross(edge_bc, edge_ab))
        cross_c = vector_lengths(-np.cross(edge_ca, edge_bc))

        self.area = cross_a / 2
        self.angle = np.vstack((np.arctan2(cross_a, dot_a), np.arctan2(cross_b, dot_b), np.arctan2(cross_c, dot_c))).T
        self.cot_angle = np.vstack((vector_safe_divide(dot_a, cross_a), vector_safe_divide(dot_b, cross_b), vector_safe_divide(dot_c, cross_c))).T


    # Computes a gradient operator for the triangle mesh.
    def compute_gradient(self):
        if self.gradient is not None:
            return

        numverts = self.verts.shape[0]
        numtris = self.triangles.shape[0]

        idx_a = self.triangles[:,0]
        idx_b = self.triangles[:,1]
        idx_c = self.triangles[:,2]

        vert_a = self.verts[idx_a,:]
        vert_b = self.verts[idx_b,:]
        vert_c = self.verts[idx_c,:]

        edge_ab = vert_b - vert_a
        edge_bc = vert_c - vert_b
        edge_ca = vert_a - vert_c

        edgenor_a = -np.cross(self.normal, edge_bc)
        edgenor_b = -np.cross(self.normal, edge_ca)
        edgenor_c = -np.cross(self.normal, edge_ab)
        gradient_a = 0.5 * vector_safe_divide(edgenor_a, self.area[:,None])
        gradient_b = 0.5 * vector_safe_divide(edgenor_b, self.area[:,None])
        gradient_c = 0.5 * vector_safe_divide(edgenor_c, self.area[:,None])

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
        G_cols = np.repeat(self.triangles, 3)
        G_data = np.hstack((gradient_a, gradient_b, gradient_c)).flatten()
        self.gradient = sparse.coo_matrix((G_data, (G_rows, G_cols)), shape=(numtris * 3, numverts))


    # Computes a divergence operator D from the gradient and triangle areas.
    def compute_divergence(self):
        if self.divergence is not None:
            return

        # TODO this could be better
        A = sparse.diags(np.repeat(self.area, 3, axis=0), 0)
        self.divergence = -self.gradient.T * A


    # Computes a Laplacian operator for the triangle mesh.
    # The Laplace operator matrix is stored as a composition of a "mass" matrix M and a "stiffness" matrix S:
    #   L = M^-1 @ S
    #
    # Parameters
    # ----------
    # vertex_stiffness : ndarray of size V and type float
    #   Stiffness factor per vertex.
    def compute_laplacian(self, vertex_stiffness : np.ndarray):
        if self.mass is not None:
            return

        numverts = self.verts.shape[0]
        numtris = self.triangles.shape[0]

        idx_a = self.triangles[:,0]
        idx_b = self.triangles[:,1]
        idx_c = self.triangles[:,2]

        vert_a = self.verts[idx_a,:]
        vert_b = self.verts[idx_b,:]
        vert_c = self.verts[idx_c,:]

        edge_ab = vert_b - vert_a
        edge_bc = vert_c - vert_b
        edge_ca = vert_a - vert_c

        # Mass contributes to each triangle corner
        mass = self.area / 12 # 1/12 contribution to each corner
        M_rows = np.r_[idx_a,  idx_a,  idx_a,  idx_b,  idx_b,  idx_b,  idx_c,  idx_c,  idx_c]
        M_cols = np.r_[idx_a,  idx_b,  idx_c,  idx_a,  idx_b,  idx_c,  idx_a,  idx_b,  idx_c]
        M_data = np.r_[2*mass, mass,   mass,   mass,   2*mass, mass,   mass,   mass,   2*mass]
        self.mass = sparse.coo_matrix((M_data, (M_rows, M_cols)), shape=(numverts, numverts))

        # Cotan stiffness matrix for triangle mesh
        stiff_a = -0.5 * self.cot_angle[:, 0] * vertex_stiffness[idx_a]
        stiff_b = -0.5 * self.cot_angle[:, 1] * vertex_stiffness[idx_b]
        stiff_c = -0.5 * self.cot_angle[:, 2] * vertex_stiffness[idx_c]
        S_rows = np.r_[idx_a,  idx_a,  idx_a,  idx_b,  idx_b,  idx_b,  idx_c,  idx_c,  idx_c]
        S_cols = np.r_[idx_a,  idx_b,  idx_c,  idx_a,  idx_b,  idx_c,  idx_a,  idx_b,  idx_c]
        S_data = np.r_[-stiff_c - stiff_b, stiff_c, stiff_b, stiff_c, -stiff_a - stiff_c, stiff_a, stiff_b, stiff_a, -stiff_b - stiff_a]
        self.stiffness = sparse.coo_matrix((S_data, (S_rows, S_cols)), shape=(numverts, numverts))


    # Compute adjacency matrix and neighborhood loops.
    def compute_neighborhood(self):
        if self.loop_verts is not None:
            return

        numverts = self.verts.shape[0]
        numtriangles = self.triangles.shape[0]
        numcorners = self.triangles.size

        # Defines adjacency for each edge.
        # For an edge from vertex i to vertex j the matrix contains next/prev corners:
        #   adjacency[i, j] = (previous corner) | (next corner) << 32
        # Entries for non-neighborhood vertices are empty/zero and must be regarded as undefined.
        cornerverts = self.triangles.flatten()
        cornerverts_prev = np.roll(self.triangles, 1, axis=1).flatten()
        cornerverts_next = np.roll(self.triangles, -1, axis=1).flatten()
        # Rows twice, one set of entries for "prev", another for "next"
        # These entries are added up by the coo_matrix constructor, so interior edges have both prev and next.
        adj_rows = np.tile(cornerverts, 2)
        adj_cols = np.r_[cornerverts_prev, cornerverts_next]
        # A triangle defines the "prev" corner of edge (i-1, i) and the "next" corner of edge (i, i+1).
        #
        #   i-1       i+1
        #    \       /
        #     \     /
        #      \ c /
        #       \ /
        #        i
        #
        corners = (np.arange(0, numtriangles * 3, 3, dtype=np.int64)[:, None] + np.array([0, 1, 2], dtype=np.int64)[None, :]).flatten()
        adj_data = np.r_[corners + 1, np.left_shift(corners + 1, 32)]
        adjacency = sparse.coo_matrix((adj_data, (adj_rows, adj_cols)), shape=(numverts, numverts))

        # Make both compressed-sparse-row and dictionary-of-keys matrices.
        # csr for counting non-zero elements and finding the start vertex.
        # dok for fast lookups when ordering corners.
        adjacency_csr = adjacency.tocsr()
        adjacency_dok = adjacency.todok()

        # Number of entries per row is size of neighborhood vertices, including boundary edges.
        loop_verts_count = adjacency_csr.getnnz(axis=0)
        loop_corners_count = np.zeros(numverts, dtype=int)
        for corner in np.nditer(cornerverts):
            loop_corners_count[corner] += 1
        # Allocate final list of neighborhood vertices, filled in below.
        self.loop_verts_index = np.r_[0, np.cumsum(loop_verts_count)]
        self.loop_corners_index = np.r_[0, np.cumsum(loop_corners_count)]
        self.loop_verts = np.empty(self.loop_verts_index[-1], dtype=int)
        self.loop_corners = np.empty(self.loop_corners_index[-1], dtype=int)

        # -1 indicates an uninitialized loop vertex ()
        adj_indptr = adjacency_csr.indptr
        adj_indices = adjacency_csr.indices
        adj_prev = np.bitwise_and(adjacency_csr.data, 0xffffffff)
        adj_next = np.right_shift(adjacency_csr.data, 32)
        # Also determine boundary verts in this loop
        self.boundary = np.full(numverts, False)
        for i in range(numverts):
            # "Seed" the loops: find boundary or use an arbitrary vertex if the loop is closed.
            start_vert = -1
            for j in range(adj_indptr[i], adj_indptr[i + 1]):
                if adj_prev[j] == 0:
                    # Use boundary vertex as start
                    start_vert = adj_indices[j]
                    self.boundary[i] = True
                    break
                if start_vert < 0:
                    start_vert = adj_indices[j]
            if start_vert < 0:
                continue

            # Build neighborhood loops by traversing the adjacency matrix row for this vertex.
            cur_vert = start_vert
            for loop_idx in range(self.loop_corners_index[i], self.loop_corners_index[i + 1]):
                self.loop_verts[loop_idx] = cur_vert

                # Loop up next neighbor.
                data = adjacency_dok[i, cur_vert]
                corner = (data >> 32) - 1
                if corner < 0:
                    # Stop when hitting boundary
                    break
                self.loop_corners[corner] = corner

                # Note: cornerverts_prev refers to ordering in the triangle, the "prev" triangle vertex is "next" for the vertex fan!
                cur_vert = cornerverts_prev[corner]
                if cur_vert == start_vert:
                    # Stop when closing the loop.
                    break


    # Compute the scaled radial angles that define local surface vector mapping.
    def compute_radial_angles(self):
        if self.radial_angle_min is not None:
            return

        numverts = self.verts.shape[0]
        numtris = self.triangles.shape[0]
        numcorners = self.triangles.size

        self.radial_angle_min = np.empty(numcorners, dtype=float)
        self.radial_angle_max = np.empty(numcorners, dtype=float)

        corner_angle = self.angle.flatten()
        for i in range(numverts):
            loop_start = self.loop_corners_index[i]
            loop_stop = self.loop_corners_index[i + 1]
            corners = self.loop_corners[loop_start:loop_stop]
            angles = corner_angle[corners]

            totangle = np.sum(angles)
            angle_scale = tau / totangle if totangle > tau or not self.boundary[i] else 1.0

            anglesum = 0.0
            for cor, ang in zip(corners, angles):
                self.radial_angle_min[cor] = anglesum
                anglesum += ang * angle_scale
                self.radial_angle_max[cor] = anglesum
