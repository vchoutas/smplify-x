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

import tqdm

try:
    input = raw_input
except NameError:
    pass

import trimesh
from fitting import MeshViewer

parser = argparse.ArgumentParser()

parser.add_argument('--mesh_fns', required=True,
                    type=str, help='The name of the result file',
                    nargs='*')

args = parser.parse_args()

input_mesh_fns = args.mesh_fns


mesh_fns = []
for mesh_fn in input_mesh_fns:
    if osp.isdir(mesh_fn):
        mesh_fns += [osp.join(root, fn)
                     for (root, dirs, files) in os.walk(mesh_fn)
                     for fn in files if fn.endswith('.obj')]
    elif osp.isfile(mesh_fn):
        mesh_fns.append(mesh_fn)

mv = MeshViewer()

for mesh_fn in tqdm.tqdm(sorted(mesh_fns)):
    out_mesh = trimesh.load(mesh_fn)

    mv.update_mesh(out_mesh.vertices, out_mesh.faces)
    input('Press any key to continue')

mv.close_viewer()
