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
import bmesh
import numpy as np
from mathutils import Color, Matrix, Vector

class VertexGroupReader:
    def __init__(self, vgroup : bpy.types.VertexGroup, default_weight):
        self.vgroup = vgroup
        self.default_weight = default_weight

    def read_scalar(self, bm):
        def values():
            dvert_lay = bm.verts.layers.deform.verify()
            idx = self.vgroup.index
            for vert in bm.verts:
                dvert = vert[dvert_lay]
                yield dvert[idx] if idx in dvert else self.default_weight
        return np.fromiter(values(), dtype=float, count=len(bm.verts))


class VertexGroupWriter:
    def __init__(self, vgroup : bpy.types.VertexGroup):
        self.vgroup = vgroup

    def write_scalar(self, bm, array):
        assert(array.size == len(bm.verts))
        dvert_lay = bm.verts.layers.deform.verify()
        idx = self.vgroup.index
        for value, vert in zip(np.nditer(array, order='C'), bm.verts):
            dvert = vert[dvert_lay]
            dvert[idx] = value


class UVLayerWriter:
    def __init__(self, uvlayer : bpy.types.MeshUVLoopLayer):
        self.uvlayer = uvlayer

    def write_scalar(self, bm, array):
        assert(array.size == len(bm.verts))
        uv_lay = bm.loops.layers.uv.get(self.uvlayer.name)
        for value, vert in zip(np.nditer(array, order='C'), bm.verts):
            for loop in vert.link_loops:
                loop[uv_lay].uv = Vector((value, 0.0))


class VertexColorWriter:
    def __init__(self, vcol : bpy.types.MeshLoopColorLayer):
        self.vcol = vcol

    def write_scalar(self, bm, array):
        assert(array.size == len(bm.verts))
        vcol_lay = bm.loops.layers.color.get(self.vcol.name)
        array_iter = np.nditer(array, order='C')
        for vert in bm.verts:
            x = next(array_iter)
            y = next(array_iter)
            z = next(array_iter)
            for loop in vert.link_loops:
                loop[vcol_lay].color = Color((x, y, z))


class TextureReader:
    def __init__(self, tex : bpy.types.Texture, mat : Matrix):
        self.tex = tex
        self.mat = mat

    def read_scalar(self, bm):
        def values():
            for vert in bm.verts:
                co = self.mat @ vert.co
                yield self.tex.evaluate(co)[0]
        return np.fromiter(values(), dtype=float, count=len(bm.verts))

    def read_vector(self, bm):
        def values():
            for vert in bm.verts:
                co = self.mat @ vert.co
                yield from self.tex.evaluate(co)[0:3]
        return np.fromiter(values(), dtype=float, count=3*len(bm.verts)).reshape((len(bm.verts), 3))


class ScaledTextureReader:
    def __init__(self, tex_reader, scale_reader):
        self.tex_reader = tex_reader
        self.scale_reader = scale_reader

    def read_vector(self, bm):
        scale = self.scale_reader.read_scalar(bm)
        tex = self.tex_reader.read_vector(bm)
        return (tex * 2.0 - 1.0) * scale[:,None]


class DualUVLayerWriter:
    def __init__(self, uvlayer_xy : bpy.types.MeshUVLoopLayer, uvlayer_z : bpy.types.MeshUVLoopLayer):
        self.uvlayer_xy = uvlayer_xy
        self.uvlayer_z = uvlayer_z

    def write_vector(self, bm, array):
        assert(array.size == 3 * len(bm.verts))
        uv_xy_lay = bm.loops.layers.uv.get(self.uvlayer_xy.name)
        uv_z_lay = bm.loops.layers.uv.get(self.uvlayer_z.name)
        array_iter = np.nditer(array, order='C')
        for vert in bm.verts:
            x = next(array_iter)
            y = next(array_iter)
            z = next(array_iter)
            for loop in vert.link_loops:
                loop[uv_xy_lay].uv = Vector((x, y))
                loop[uv_z_lay].uv = Vector((z, 0.0))
