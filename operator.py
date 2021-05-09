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
from .layers import *
from .heat_map import HeatMapGenerator


# data_type in { 'SCALAR', 'VECTOR' }
def _make_layer_settings(default_name, data_type):
    scalar_layer_types_items = [
        ('VERTEX_GROUP', "Vertex Group", "Vertex group"),
        ('VERTEX_COLOR', "Vertex Color", "Vertex color layer"),
        ('UV_LAYER', "UV Layer", "UV Layer"),
    ]
    scalar_layer_types_default = {'VERTEX_GROUP', 'UV_LAYER'}
    vector_layer_types_items = [
        ('VERTEX_COLOR', "Vertex Color", "Vertex color layer"),
    ]
    vector_layer_types_default = {'VERTEX_COLOR'}

    layer_types_map = {
        'SCALAR' : (scalar_layer_types_items, scalar_layer_types_default),
        'VECTOR' : (vector_layer_types_items, vector_layer_types_default),
    }
    layer_types_items, layer_types_default = layer_types_map[data_type]

    class OutputLayerSettings(bpy.types.PropertyGroup):
        layer_name : StringProperty(
            name="Layer Name",
            description="Name of the vertex data layer",
            default=default_name,
        )

        layer_types : EnumProperty(
            items=layer_types_items,
            name="Layer Types",
            description="Types of vertex data layers to use for storing values",
            default=layer_types_default,
            options={'ENUM_FLAG'},
        )

    return OutputLayerSettings


def _get_layer_writer(settings, obj):
    if 'VERTEX_GROUP' in settings.layer_types:
        vgroup = obj.vertex_groups.get(settings.layer_name, None)
        if vgroup is None:
            vgroup = obj.vertex_groups.new(name=settings.layer_name)
        vgroup_writer = VertexGroupWriter(vgroup)
    else:
        vgroup_writer = None

    if 'UV_LAYER' in settings.layer_types:
        uvlayer = obj.data.uv_layers.get(settings.layer_name, None)
        if uvlayer is None:
            uvlayer = obj.data.uv_layers.new(name=settings.layer_name, do_init=False)
        uvlayer_writer = UVLayerWriter(uvlayer)
    else:
        uvlayer_writer = None

    if 'VERTEX_COLOR' in settings.layer_types:
        vcol = obj.data.vertex_colors.get(settings.layer_name, None)
        if vcol is None:
            vcol = obj.data.vertex_colors.new(name=settings.layer_name, do_init=False)
        vcol_writer = VertexColorWriter(vcol)
    else:
        vcol_writer = None

    def combined_writer(bm, array):
        if vgroup_writer:
            vgroup_writer(bm, array)
        if uvlayer_writer:
            uvlayer_writer(bm, array)
        if vcol_writer:
            vcol_writer(bm, array)

    return combined_writer


HeatOutputLayerSettings = _make_layer_settings("Heat", 'SCALAR')
DistanceOutputLayerSettings = _make_layer_settings("Distance", 'SCALAR')


class GeodesicDistanceOperator(bpy.types.Operator):
    """Generate mesh attributes for geodesic distance."""
    bl_idname = 'mesh.geodesic_distance'
    bl_label = 'Geodesic Distance'
    bl_options = {'UNDO', 'REGISTER'}

    boundary_vgroup : StringProperty(
        name="Boundary",
        description="Vertex group that defines the boundary where geodesic distance is zero",
        default="Boundary",
    )

    heat_output_layers : PointerProperty(type=HeatOutputLayerSettings)
    distance_output_layers : PointerProperty(type=DistanceOutputLayerSettings)

    heat_time_scale : FloatProperty(
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

        orig_mode = obj.mode
        bpy.ops.object.mode_set(mode='OBJECT')
        try:
            # Note: Create writers before bm.from_mesh, so data layers are fully initialized
            heat_writer = _get_layer_writer(self.heat_output_layers, obj)
            distance_writer = _get_layer_writer(self.distance_output_layers, obj)

            bm = bmesh.new()
            bm.from_mesh(obj.data)

            heat_map_gen = HeatMapGenerator(bm)
            heat_map_gen.generate(
                VertexGroupReader(boundary_vg, 0.0),
                heat_writer,
                distance_writer,
                self.heat_time_scale
            )

            bm.to_mesh(obj.data)
            bm.free()
            obj.data.update()

        finally:
            bpy.ops.object.mode_set(mode=orig_mode)

        return {'FINISHED'}

def menu_func(self, context):
    self.layout.separator()
    props = self.layout.operator(GeodesicDistanceOperator.bl_idname)


def register():
    bpy.utils.register_class(HeatOutputLayerSettings)
    bpy.utils.register_class(DistanceOutputLayerSettings)
    bpy.utils.register_class(GeodesicDistanceOperator)
    bpy.types.MESH_MT_vertex_group_context_menu.append(menu_func)
    bpy.types.VIEW3D_MT_paint_weight.append(menu_func)
    bpy.types.VIEW3D_MT_object.append(menu_func)

def unregister():
    bpy.utils.unregister_class(HeatOutputLayerSettings)
    bpy.utils.unregister_class(DistanceOutputLayerSettings)
    bpy.utils.unregister_class(GeodesicDistanceOperator)
    bpy.types.MESH_MT_vertex_group_context_menu.remove(menu_func)
    bpy.types.VIEW3D_MT_paint_weight.remove(menu_func)
    bpy.types.VIEW3D_MT_object.remove(menu_func)
