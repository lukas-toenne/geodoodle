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
from bpy.props import *
from .heat_map import HeatMapGenerator


class GeodesicDistanceOperator(bpy.types.Operator):
    """Generate mesh attributes for geodesic distance."""
    bl_idname = 'mesh.geodesic_distance'
    bl_label = 'Geodesic Distance'
    bl_options = {'UNDO', 'REGISTER'}

    heat_vgroup : StringProperty(
        name="Heat",
        description="Output vertex group to write heat values into",
        default="Heat",
    )

    distance_vgroup : StringProperty(
        name="Distance",
        description="Output vertex group to write distance values into",
        default="Distance",
    )

    boundary_vgroup : StringProperty(
        name="Boundary",
        description="Vertex group that defines the boundary where geodesic distance is zero",
        default="Boundary",
    )

    time_scale : FloatProperty(
        name="Time Scale",
        description="Scaling of the time step for heat map calculation",
        default=1.0,
        soft_min=0.1,
        soft_max=10.0,
        min=0.0001,
        max=10000.0,
        subtype='FACTOR',
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH'

    def execute(self, context):
        obj = context.active_object

        def ensure_vgroup(name):
            vg = obj.vertex_groups.get(name, None)
            if vg is None:
                vg = obj.vertex_groups.new(name=name)
            return vg

        boundary_vg = obj.vertex_groups.get(self.boundary_vgroup, None)
        if boundary_vg is None:
            self.report({'ERROR_INVALID_CONTEXT'}, "Object is missing boundary vertex group " + self.boundary_vgroup)
            return {'CANCELLED'}

        heat_vg = ensure_vgroup(self.heat_vgroup)
        if heat_vg is None:
            self.report({'ERROR'}, "Could not create heat vertex group " + self.heat_vgroup)
            return {'CANCELLED'}

        distance_vg = ensure_vgroup(self.distance_vgroup)
        if distance_vg is None:
            self.report({'ERROR'}, "Could not create distance vertex group " + self.distance_vgroup)
            return {'CANCELLED'}

        orig_mode = obj.mode
        bpy.ops.object.mode_set(mode='OBJECT')
        try:
            bm = bmesh.new()
            bm.from_mesh(obj.data)

            heat_map_gen = HeatMapGenerator(bm)
            heat_map_gen.generate(boundary_vg, heat_vg, self.time_scale)

            bm.to_mesh(obj.data)
            bm.free()
            obj.data.update()

        finally:
            bpy.ops.object.mode_set(mode=orig_mode)

        return {'FINISHED'}

def menu_func(self, context):
    self.layout.separator()
    self.layout.operator(GeodesicDistanceOperator.bl_idname)


def register():
    bpy.utils.register_class(GeodesicDistanceOperator)
    bpy.types.MESH_MT_vertex_group_context_menu.append(menu_func)
    bpy.types.VIEW3D_MT_paint_weight.append(menu_func)

def unregister():
    bpy.utils.unregister_class(GeodesicDistanceOperator)
    bpy.types.MESH_MT_vertex_group_context_menu.remove(menu_func)
    bpy.types.VIEW3D_MT_paint_weight.remove(menu_func)
