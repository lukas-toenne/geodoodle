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

def log_matrix(m, name):
    if isinstance(m, np.ndarray):
        size = m.nbytes
        mtype = "numpy"
    else:
        size = m.data.size
        mtype = "scipy"
    if size > 1024.0:
        size /= 1024.0
        if size > 1024.0:
            size /= 1024.0
            unit = "MB"
        else:
            unit = "KB"
    else:
        unit = "Bytes"
    print("{name: <4} {mtype: <8} {shape: <20} {size:.2f} {unit}".format(name=name+":", shape=str(m.shape), mtype="["+mtype+"]", size=size, unit=unit))
    pass


def debug_build_virtual_verts(bm, virtual_verts):
    assert(len(virtual_verts) == len(bm.faces))
    old_faces = list(bm.faces)
    for oface, vv in zip(old_faces, virtual_verts):
        nvert = bm.verts.new(vv)
        for loop in oface.loops:
            bm.faces.new((nvert, loop.vert, loop.link_loop_next.vert))
        bm.faces.remove(oface)
