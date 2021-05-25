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
import time
from bpy.props import *
from contextlib import contextmanager
from .layers import *
from . import geometry_math


# data_type in { 'SCALAR', 'VECTOR' }
def _make_output_layer_settings(default_name, data_type):
    scalar_layer_types_items = [
        ('VERTEX_GROUP', "Vertex Group", "Vertex group"),
        ('VERTEX_COLOR', "Vertex Color", "Vertex color layer"),
        ('UV_LAYER', "UV Layer", "UV Layer"),
    ]
    scalar_layer_types_default = {'VERTEX_GROUP', 'UV_LAYER'}
    vector_layer_types_items = [
        ('UV_LAYER', "UV Layer", "UV Layer"),
        ('VERTEX_COLOR', "Vertex Color", "Vertex color layer"),
    ]
    vector_layer_types_default = {'UV_LAYER'}

    def scalar_writers(settings, obj):
        if 'VERTEX_GROUP' in settings.layer_types:
            vgroup = obj.vertex_groups.get(settings.layer_name, None)
            if vgroup is None:
                vgroup = obj.vertex_groups.new(name=settings.layer_name)
            yield VertexGroupWriter(vgroup)

        if 'UV_LAYER' in settings.layer_types:
            uvlayer = obj.data.uv_layers.get(settings.layer_name, None)
            if uvlayer is None:
                uvlayer = obj.data.uv_layers.new(name=settings.layer_name, do_init=False)
            yield UVLayerWriter(uvlayer)

        if 'VERTEX_COLOR' in settings.layer_types:
            vcol = obj.data.vertex_colors.get(settings.layer_name, None)
            if vcol is None:
                vcol = obj.data.vertex_colors.new(name=settings.layer_name, do_init=False)
            yield VertexColorWriter(vcol)

    def vector_writers(settings, obj):
        if 'VERTEX_GROUP' in settings.layer_types:
            pass

        if 'UV_LAYER' in settings.layer_types:
            uvlayer_xy = obj.data.uv_layers.get(settings.layer_name + ".xy", None)
            uvlayer_z = obj.data.uv_layers.get(settings.layer_name + ".z", None)
            if uvlayer_xy is None:
                uvlayer_xy = obj.data.uv_layers.new(name=settings.layer_name + ".xy", do_init=False)
            if uvlayer_z is None:
                uvlayer_z = obj.data.uv_layers.new(name=settings.layer_name + ".z", do_init=False)
            yield DualUVLayerWriter(uvlayer_xy, uvlayer_z)

        if 'VERTEX_COLOR' in settings.layer_types:
            vcol = obj.data.vertex_colors.get(settings.layer_name, None)
            if vcol is None:
                vcol = obj.data.vertex_colors.new(name=settings.layer_name, do_init=False)
            yield VertexColorWriter(vcol)

    class ScalarLayerWriter:
        def __init__(self, settings, obj):
            self.writers = list(scalar_writers(settings, obj))

        def write_scalar(self, bm, array):
            for w in self.writers:
                w.write_scalar(bm, array)

    class VectorLayerWriter:
        def __init__(self, settings, obj):
            self.writers = list(vector_writers(settings, obj))

        def write_vector(self, bm, array):
            for w in self.writers:
                w.write_vector(bm, array)

    layer_types_map = {
        'SCALAR' : (scalar_layer_types_items, scalar_layer_types_default, ScalarLayerWriter),
        'VECTOR' : (vector_layer_types_items, vector_layer_types_default, VectorLayerWriter),
    }
    layer_types_items, layer_types_default, layer_writer_cls = layer_types_map[data_type]

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

        def draw(self, context, layout, text):
            layout.label(text=text)
            layout.prop(self, "layer_name")
            row = layout.row(align=True)
            row.prop(self, "layer_types", expand=True)

        def get_writer(self, obj):
            return layer_writer_cls(self, obj)

    return OutputLayerSettings


HeatOutputLayerSettings = _make_output_layer_settings("Heat", 'SCALAR')
DistanceOutputLayerSettings = _make_output_layer_settings("Distance", 'SCALAR')
VectorOutputLayerSettings = _make_output_layer_settings("Vector", 'VECTOR')


class GeoDoodleOperatorBase(bpy.types.Operator):
    @staticmethod
    def ensure_vgroup(obj, name):
        vg = obj.vertex_groups.get(name, None)
        if vg is None:
            vg = obj.vertex_groups.new(name=name)
        return vg

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH'

    def execute(self, context):
        obj = context.active_object

        orig_mode = obj.mode
        bpy.ops.object.mode_set(mode='OBJECT')
        try:
            # Note: Creates writers before bm.from_mesh, so data layers are fully initialized
            with self.get_generator(obj) as generator:
                depsgraph = context.evaluated_depsgraph_get()
                bm = bmesh.new()
                bm.from_object(obj, depsgraph)
                generator(bm)
                bm.to_mesh(obj.data)
                bm.free()
            obj.data.update()
        finally:
            bpy.ops.object.mode_set(mode=orig_mode)

        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class HeatMapOperator(GeoDoodleOperatorBase):
    """Generate mesh attributes for a heat map."""
    bl_idname = 'geodoodle.heat_map'
    bl_label = 'Heat Map'
    bl_options = {'UNDO', 'REGISTER'}

    source_vgroup : StringProperty(
        name="Source",
        description="Vertex group that defines the source where geodesic distance is zero",
        default="Source",
    )

    obstacle_vgroup : StringProperty(
        name="Obstacle",
        description="Vertex group that artificially locally decreases geodesic distance",
        default="Obstacle",
    )

    heat_output_layers : PointerProperty(type=HeatOutputLayerSettings)

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

    @contextmanager
    def get_generator(self, obj):
        source_vg = obj.vertex_groups.get(self.source_vgroup, None)
        if source_vg is None:
            self.report({'ERROR_INVALID_CONTEXT'}, "Object is missing source vertex group " + self.source_vgroup)
            return {'CANCELLED'}

        obstacle_vg = obj.vertex_groups.get(self.obstacle_vgroup, None)

        source_reader = VertexGroupReader(source_vg, 0.0)
        obstacle_reader = VertexGroupReader(obstacle_vg, 0.0) if obstacle_vg else None
        heat_writer = self.heat_output_layers.get_writer(obj)

        def apply(bm):
            perf_start = time.perf_counter()
            geometry_math.compute_heat(bm, source_reader, obstacle_reader, heat_writer, self.heat_time_scale)
            perf_end = time.perf_counter()
            print("Heat map computation time: {:0.4f}".format(perf_end - perf_start))

        yield apply

    def draw(self, context):
        layout = self.layout

        layout.prop(self, "source_vgroup")
        layout.prop(self, "obstacle_vgroup")

        box = layout.box()
        self.heat_output_layers.draw(context, box, text="Heat Output:")
        box.prop(self, "heat_time_scale")


class GeodesicDistanceOperator(GeoDoodleOperatorBase):
    """Generate mesh attributes for geodesic distance."""
    bl_idname = 'geodoodle.geodesic_distance'
    bl_label = 'Geodesic Distance'
    bl_options = {'UNDO', 'REGISTER'}

    source_vgroup : StringProperty(
        name="Source",
        description="Vertex group that defines the source where geodesic distance is zero",
        default="Source",
    )

    obstacle_vgroup : StringProperty(
        name="Obstacle",
        description="Vertex group that artificially locally decreases geodesic distance",
        default="Obstacle",
    )

    distance_output_layers : PointerProperty(type=DistanceOutputLayerSettings)

    @contextmanager
    def get_generator(self, obj):
        source_vg = obj.vertex_groups.get(self.source_vgroup, None)
        if source_vg is None:
            self.report({'ERROR_INVALID_CONTEXT'}, "Object is missing source vertex group " + self.source_vgroup)
            return {'CANCELLED'}

        obstacle_vg = obj.vertex_groups.get(self.obstacle_vgroup, None)

        source_reader = VertexGroupReader(source_vg, 0.0)
        obstacle_reader = VertexGroupReader(obstacle_vg, 0.0) if obstacle_vg else None
        distance_writer = self.distance_output_layers.get_writer(obj)

        def apply(bm):
            perf_start = time.perf_counter()
            geometry_math.compute_distance(bm, source_reader, obstacle_reader, distance_writer)
            perf_end = time.perf_counter()
            print("Geodesic distance computation time: {:0.4f}".format(perf_end - perf_start))

        yield apply

    def draw(self, context):
        layout = self.layout

        layout.prop(self, "source_vgroup")
        layout.prop(self, "obstacle_vgroup")

        box = layout.box()
        self.distance_output_layers.draw(context, box, text="Distance Output:")


class ParallelTransportOperator(GeoDoodleOperatorBase):
    """Generate mesh attributes for parallel transport of a vector field."""
    bl_idname = 'geodoodle.parallel_transport'
    bl_label = 'Parallel Transport'
    bl_options = {'UNDO', 'REGISTER'}

    source_vgroup : StringProperty(
        name="Source",
        description="Vertex group that defines the source where geodesic distance is zero",
        default="Source",
    )

    obstacle_vgroup : StringProperty(
        name="Obstacle",
        description="Vertex group that artificially locally decreases geodesic distance",
        default="Obstacle",
    )

    vector_texture : StringProperty(
        name="Vector Texture",
        description="Texture name for reading as a vector field (PLACEHOLDER)",
        default="Texture",
    )

    vector_output_layers : PointerProperty(type=VectorOutputLayerSettings)

    @contextmanager
    def get_generator(self, obj):
        source_vg = obj.vertex_groups.get(self.source_vgroup, None)
        if source_vg is None:
            self.report({'ERROR_INVALID_CONTEXT'}, "Object is missing source vertex group " + self.source_vgroup)
            return {'CANCELLED'}

        obstacle_vg = obj.vertex_groups.get(self.obstacle_vgroup, None)

        source_reader = VertexGroupReader(source_vg, 0.0)
        obstacle_reader = VertexGroupReader(obstacle_vg, 0.0) if obstacle_vg else None
        tex = bpy.data.textures.get(self.vector_texture, None)
        mat = Matrix.Identity(4)
        tex_reader = TextureReader(tex, mat) if tex else None
        vector_reader = ScaledTextureReader(tex_reader, source_reader)
        vector_writer = self.vector_output_layers.get_writer(obj)

        def apply(bm):
            perf_start = time.perf_counter()
            geometry_math.compute_parallel_transport(bm, vector_reader, obstacle_reader, vector_writer)
            perf_end = time.perf_counter()
            print("Parallel transport computation time: {:0.4f}".format(perf_end - perf_start))

        yield apply

    def draw(self, context):
        layout = self.layout

        layout.prop(self, "source_vgroup")
        layout.prop(self, "obstacle_vgroup")
        layout.prop(self, "vector_texture")

        box = layout.box()
        self.vector_output_layers.draw(context, box, text="Vector Output:")


def menu_func(self, context):
    self.layout.separator()
    props = self.layout.operator(HeatMapOperator.bl_idname)
    props = self.layout.operator(GeodesicDistanceOperator.bl_idname)
    props = self.layout.operator(ParallelTransportOperator.bl_idname)


def register():
    bpy.utils.register_class(HeatOutputLayerSettings)
    bpy.utils.register_class(DistanceOutputLayerSettings)
    bpy.utils.register_class(VectorOutputLayerSettings)

    bpy.utils.register_class(HeatMapOperator)
    bpy.utils.register_class(GeodesicDistanceOperator)
    bpy.utils.register_class(ParallelTransportOperator)

    bpy.types.MESH_MT_vertex_group_context_menu.append(menu_func)
    bpy.types.VIEW3D_MT_paint_weight.append(menu_func)
    bpy.types.VIEW3D_MT_object.append(menu_func)

def unregister():
    bpy.utils.unregister_class(HeatOutputLayerSettings)
    bpy.utils.unregister_class(DistanceOutputLayerSettings)
    bpy.utils.unregister_class(VectorOutputLayerSettings)

    bpy.utils.unregister_class(HeatMapOperator)
    bpy.utils.unregister_class(GeodesicDistanceOperator)
    bpy.utils.unregister_class(ParallelTransportOperator)

    bpy.types.MESH_MT_vertex_group_context_menu.remove(menu_func)
    bpy.types.VIEW3D_MT_paint_weight.remove(menu_func)
    bpy.types.VIEW3D_MT_object.remove(menu_func)
