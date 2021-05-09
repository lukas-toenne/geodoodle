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
    "name": "GeoDoozy",
    "author": "Lukas Toenne",
    "version": (0, 1),
    "blender": (2, 92, 0),
    "location": "",
    "description": "Compute geodesic distance using a heat method",
    "warning": "",
    "doc_url": "",
    "category": "Mesh",
}

import bpy
from . import heat_map, layers, operator

if "bpy" in locals():
    import importlib
    importlib.reload(heat_map)
    importlib.reload(layers)
    importlib.reload(operator)


# XXX run this to install scipy!
# https://blender.stackexchange.com/a/153520
def install_scipy():
    import sys
    import subprocess
    import bpy

    py_exec = sys.executable
    py_prefix = sys.exec_prefix
    # ensure pip is installed & update
    subprocess.call([str(py_exec), "-m", "ensurepip", "--user"])
    subprocess.call([str(py_exec), "-m", "pip", "install", "--target={}".format(py_prefix), "--upgrade", "pip"])
    # install dependencies using pip
    # dependencies such as 'numpy' could be added to the end of this command's list
    subprocess.call([str(py_exec),"-m", "pip", "install", "--target={}".format(py_prefix), "scipy"])

def register():
    # install_scipy()
    operator.register()

def unregister():
    operator.unregister()

if __name__ == "__main__":
    register()
