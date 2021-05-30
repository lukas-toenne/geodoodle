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


# Connected neighborhood around the vertex following same ordering as loops.
def vertex_neighbors(vert):
    for loop in vert.link_loops:
        yield loop.link_loop_next.vert

        # Also yield the opposing edge in case of boundary edges
        prev_loop = loop.link_loop_prev
        if prev_loop.link_loop_radial_next == prev_loop:
            yield prev_loop.vert
    # for edge in vert.link_edges:
    #     yield edge.other_vert(vert)


# Project 3D vectors onto the surface using vertex normals.
# Returns array of 2 surface vector components and the scalar normal component.
def project_vectors_on_surface(bm, vectors):
    def values():
        vec_iter = np.nditer(vectors, order='C')
        for vert in bm.verts:
            vec = Vector((next(vec_iter), next(vec_iter), next(vec_iter)))

            if not vert.link_edges:
                yield from [0.0, 0.0, 0.0]
                continue

            # Simple projection onto the vertex normal plane.
            # Angle is relative to the first edge projected onto the same plane.
            axis_x = vert.link_edges[0].other_vert(vert).co - vert.co
            axis_x = axis_x - vert.normal * axis_x.dot(vert.normal)
            axis_x.normalize()

            axis_y = vert.normal.cross(axis_x)

            yield vec.dot(axis_x)
            yield vec.dot(axis_y)
            yield vec.dot(vert.normal)

    totverts = len(bm.verts)
    assert(vectors.size == 3 * totverts)
    local_vectors = np.fromiter((v for v in values()), dtype=float, count=totverts * 3).reshape((totverts, 3))
    return np.vectorize(complex)(local_vectors[:,0], local_vectors[:,1]), local_vectors[:,2]


# Compute 3D vectors from surface vectors.
# Surface vectors are given as complex numbers, relative to local vertex neighborhood.
# Offset numbers are scalar values in the vertex normal direction.
def surface_vectors_to_world(bm, surface_vectors, normal_components):
    def values():
        real_iter = np.nditer(np.real(surface_vectors), order='C')
        imag_iter = np.nditer(np.imag(surface_vectors), order='C')
        normal_iter = np.nditer(normal_components, order='C')
        for vert in bm.verts:
            re = next(real_iter)
            im = next(imag_iter)
            no = next(normal_iter)

            if not vert.link_edges:
                yield from [0.0, 0.0, 0.0]
                continue

            # Simple projection onto the vertex normal plane.
            # Angle is relative to the first edge projected onto the same plane.
            axis_x = vert.link_edges[0].other_vert(vert).co - vert.co
            axis_x = axis_x - vert.normal * axis_x.dot(vert.normal)
            axis_x.normalize()

            axis_y = vert.normal.cross(axis_x)

            vec = axis_x * re + axis_y * im + vert.normal * no

            yield from vec[:]

    totverts = len(bm.verts)
    assert(surface_vectors.size == totverts)
    assert(normal_components.size == totverts)
    return np.fromiter((v for v in values()), dtype=float, count=totverts * 3).reshape((totverts, 3))

# def surface_vectors_to_world(bm, surface_vectors):
#     totverts = len(bm.verts)
#     totloops = sum(len(v.link_loops) for v in bm.verts)
#     assert(surface_vectors.size == totverts)

#     bm.verts.index_update() # TODO FOR DEBUGGING ONLY
#     def values():
#         angles = np.angle(surface_vectors)
#         lengths = np.absolute(surface_vectors)

#         angle_iter = np.nditer(angles, order='C')
#         length_iter = np.nditer(lengths, order='C')
#         for vert in bm.verts:
#             numedges = len(vert.link_edges)
#             for v in vertex_neighbors(vert):
#                 ne = v.co - vert.co
#                 a = atan2(ne[1], ne[0])
#                 print("  {}: {} | ({:.3f}, {:.3f})".format(v.index, a, ne[0], ne[1]))
#             neighbor_edges = np.fromiter((c for v in vertex_neighbors(vert) for c in (v.co - vert.co)[:]), dtype=float, count=numedges * 3).reshape((numedges, 3))
#             neighbor_edge_lengths = np.linalg.norm(neighbor_edges, axis=1)
#             neighbor_edges = np.divide(neighbor_edges, neighbor_edge_lengths[:,None], out=np.zeros_like(neighbor_edges), where=neighbor_edge_lengths[:,None] != 0)
#             neighbor_next = np.roll(neighbor_edges, shift=-1, axis=0)

#             loop_angle_sin = np.linalg.norm(np.cross(neighbor_edges, neighbor_next), axis=1)
#             loop_angle_cos = np.sum(neighbor_edges * neighbor_next, axis=1)
#             loop_angles_raw = np.arctan2(loop_angle_sin, loop_angle_cos)
#             totangle = np.sum(loop_angles_raw)
#             loop_angles = loop_angles_raw / totangle * 2*pi if totangle != 0 else np.zeros_like(loop_angles_raw)
#             loop_angle_max = np.cumsum(loop_angles)
#             loop_angle_min = np.concatenate(([0.0], loop_angle_max[:-1]))

#             angle = next(angle_iter)
#             angle = angle if angle > 0.0 else angle + 2*pi
#             length = next(length_iter)

#             active_loop_index = np.argwhere(np.logical_and(loop_angle_min <= angle, loop_angle_max > angle)).flatten()
#             active_index = active_loop_index[0] if active_loop_index.size > 0 else -1
#             angle_min = loop_angle_min[active_index]
#             angle_max = loop_angle_max[active_index]
#             raw_angle = loop_angles_raw[active_index]
#             raw_angle_sin = loop_angle_sin[active_index]
#             edge_min = neighbor_edges[active_index]
#             edge_max = neighbor_next[active_index]

#             # Slerp
#             t = (angle - angle_min) / (angle_max - angle_min)
#             print("-----------------")
#             print(loop_angle_min)
#             print(loop_angle_max)
#             print(active_loop_index, active_index, "t({} <= {} < {}) = {}".format(angle_min, angle, angle_max, t))
#             # yield from (sin((1.0 - t) * raw_angle) * edge_min + sin(t * raw_angle) * edge_max) / raw_angle_sin * length
#             # yield from ((1.0 - t) * edge_min + t * edge_max) * length
#             yield from edge_min

#     return np.fromiter(values(), dtype=float, count=3 * totverts).reshape(totverts, 3)
