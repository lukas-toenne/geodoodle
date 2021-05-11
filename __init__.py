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

bl_info = {
    "name": "GeoDoodle",
    "author": "Lukas Toenne",
    "version": (0, 1),
    "blender": (2, 92, 0),
    "location": "View3D > Object > Geodesic Distance",
    "description": "Compute geodesic distance using a heat method",
    "warning": "",
    "doc_url": "https://github.com/lukas-toenne/geodoodle",
    "tracker_url": "https://github.com/lukas-toenne/geodoodle/issues",
    "category": "Mesh",
}

import bpy
from . import util, heat_map, heat_map_old, layers, operator

if "bpy" in locals():
    import importlib
    importlib.reload(util)
    importlib.reload(heat_map)
    importlib.reload(heat_map_old)
    importlib.reload(layers)
    importlib.reload(operator)


scipy_found = False
def check_scipy():
    global scipy_found
    import importlib
    from importlib import util
    scipy_spec = importlib.util.find_spec("scipy")
    scipy_found = scipy_spec is not None

check_scipy()

class SciPyInstallOperator(bpy.types.Operator):
    """Install SciPy into local Blender python packages"""
    bl_idname = "geodoodle.scipy_install"
    bl_label = "Install SciPy"
    bl_options = {'REGISTER'}

    def execute(self, context):
        import sys
        import subprocess

        # https://blender.stackexchange.com/a/153520
        py_exec = sys.executable
        py_prefix = sys.exec_prefix
        # ensure pip is installed & update
        subprocess.call([str(py_exec), "-m", "ensurepip", "--user"])
        subprocess.call([str(py_exec), "-m", "pip", "install", "--target={}".format(py_prefix), "--upgrade", "pip"])
        # install dependencies using pip
        # dependencies such as 'numpy' could be added to the end of this command's list
        subprocess.call([str(py_exec),"-m", "pip", "install", "--target={}".format(py_prefix), "scipy"])

        check_scipy()

        return {'FINISHED'}


class GeoDoodleAddonPreferences(bpy.types.AddonPreferences):
    # this must match the add-on name, use '__package__'
    # when defining this in a submodule of a python package.
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        if not scipy_found:
            layout.label(text="SciPy not installed in local Blender packages", icon='ERROR')
        layout.operator(SciPyInstallOperator.bl_idname)


def register():
    operator.register()
    bpy.utils.register_class(SciPyInstallOperator)
    bpy.utils.register_class(GeoDoodleAddonPreferences)

def unregister():
    operator.unregister()
    bpy.utils.unregister_class(SciPyInstallOperator)
    bpy.utils.unregister_class(GeoDoodleAddonPreferences)

if __name__ == "__main__":
    register()
