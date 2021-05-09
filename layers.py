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
from mathutils import Color, Vector

class VertexGroupReader:
    def __init__(self, vgroup : bpy.types.VertexGroup, default_weight):
        self.vgroup = vgroup
        self.default_weight = default_weight

    def __call__(self, bm):
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

    def __call__(self, bm, array):
        assert(array.size == len(bm.verts))
        dvert_lay = bm.verts.layers.deform.verify()
        idx = self.vgroup.index
        for value, vert in zip(np.nditer(array), bm.verts):
            dvert = vert[dvert_lay]
            dvert[idx] = value


class UVLayerWriter:
    def __init__(self, uvlayer : bpy.types.MeshUVLoopLayer):
        self.uvlayer = uvlayer

    def __call__(self, bm, array):
        assert(array.size == len(bm.verts))
        uv_lay = bm.loops.layers.uv.get(self.uvlayer.name)
        for value, vert in zip(np.nditer(array), bm.verts):
            for loop in vert.link_loops:
                loop[uv_lay].uv = Vector((value, 0.0))


class VertexColorWriter:
    def __init__(self, vcol : bpy.types.MeshLoopColorLayer):
        self.vcol = vcol

    def __call__(self, bm, array):
        print(array.shape, len(bm.verts), 3 * len(bm.verts))
        assert(array.size == 3 * len(bm.verts))
        vcol_lay = bm.loops.layers.color.get(self.vcol.name)
        array_iter = np.nditer(array)
        for vert in bm.verts:
            x = next(array_iter)
            y = next(array_iter)
            z = next(array_iter)
            for loop in vert.link_loops:
                loop[vcol_lay].color = Color((x, y, z))
