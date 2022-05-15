# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import argparse

import time
import trimesh
from mesh_viewer import MeshViewer


class KeyHandler(object):
    def __init__(self, mesh_fns, verbose=False):
        self.mesh_fns = mesh_fns
        self.idx = 0
        self.verbose = verbose
        self.close = False

    def next_mesh(self, viewer):
        self.idx += 1
        self.idx = self.idx % len(self.mesh_fns)

        if self.verbose:
            print('Loading {} ...'.format(self.mesh_fns[self.idx]))

    def prev_mesh(self, viewer):
        self.idx -= 1
        self.idx = self.idx % len(self.mesh_fns)

        if self.verbose:
            print('Loading {} ...'.format(self.mesh_fns[self.idx]))

    def get_mesh_fn(self):
        return self.mesh_fns[self.idx]

    def quit_viewer(self, viewer):
        self.close = True


parser = argparse.ArgumentParser()

parser.add_argument('--mesh_fns', required=True,
                    type=str, help='The name of the result file',
                    nargs='*')
parser.add_argument('--verbose', action='store_true',
                    help='Verbosity flag')

args = parser.parse_args()

input_mesh_fns = args.mesh_fns
verbose = args.verbose

mesh_fns = []
for mesh_fn in input_mesh_fns:
    if osp.isdir(mesh_fn):
        mesh_fns += [osp.join(root, fn)
                     for (root, dirs, files) in os.walk(mesh_fn)
                     for fn in files if fn.endswith('.obj')]
    elif osp.isfile(mesh_fn):
        mesh_fns.append(mesh_fn)
mesh_fns.sort()

key_handler = KeyHandler(mesh_fns)
registered_keys = {'q': key_handler.quit_viewer,
                   '+': key_handler.next_mesh, '-': key_handler.prev_mesh}
mv = MeshViewer(registered_keys=registered_keys)

print('Press q to exit')
print('Press + to open next mesh')
print('Press - to open previous mesh')

close = False
while True:
    if not mv.is_active():
        break
    if key_handler.close:
        break

    mesh_fn = key_handler.get_mesh_fn()
    #  if prev_idx == idx:
    #  continue
    out_mesh = trimesh.load(mesh_fn)

    mv.update_mesh(out_mesh.vertices, out_mesh.faces)
    time.sleep(0.1)

mv.close_viewer()
