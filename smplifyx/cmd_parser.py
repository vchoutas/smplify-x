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
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import configargparse


def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter

    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'PyTorch implementation of SMPLifyX'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='SMPLifyX')

    parser.add_argument('--data_folder',
                        default=os.getcwd(),
                        help='The directory that contains the data.')
    parser.add_argument('--max_persons', type=int, default=3,
                        help='The maximum number of persons to process')
    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='config file path')
    parser.add_argument('--loss_type', default='smplify', type=str,
                        help='The type of loss to use')
    parser.add_argument('--interactive',
                        type=lambda arg: arg.lower() == 'true',
                        default=False,
                        help='Print info messages during the process')
    parser.add_argument('--save_meshes',
                        type=lambda arg: arg.lower() == 'true',
                        default=True,
                        help='Save final output meshes')
    parser.add_argument('--visualize',
                        type=lambda arg: arg.lower() == 'true',
                        default=False,
                        help='Display plots while running the optimization')
    parser.add_argument('--degrees', type=float, default=[0, 90, 180, 270],
                        nargs='*',
                        help='Degrees of rotation for rendering the final' +
                        ' result')
    parser.add_argument('--use_cuda',
                        type=lambda arg: arg.lower() == 'true',
                        default=True,
                        help='Use CUDA for the computations')
    parser.add_argument('--dataset', default='hands_cmu_gt', type=str,
                        help='The name of the dataset that will be used')
    parser.add_argument('--joints_to_ign', default=-1, type=int,
                        nargs='*',
                        help='Indices of joints to be ignored')
    parser.add_argument('--output_folder',
                        default='output',
                        type=str,
                        help='The folder where the output is stored')
    parser.add_argument('--img_folder', type=str, default='images',
                        help='The folder where the images are stored')
    parser.add_argument('--keyp_folder', type=str, default='keypoints',
                        help='The folder where the keypoints are stored')
    parser.add_argument('--summary_folder', type=str, default='summaries',
                        help='Where to store the TensorBoard summaries')
    parser.add_argument('--result_folder', type=str, default='results',
                        help='The folder with the pkls of the output' +
                        ' parameters')
    parser.add_argument('--mesh_folder', type=str, default='meshes',
                        help='The folder where the output meshes are stored')
    parser.add_argument('--gender_lbl_type', default='none',
                        choices=['none', 'gt', 'pd'], type=str,
                        help='The type of gender label to use')
    parser.add_argument('--gender', type=str,
                        default='neutral',
                        choices=['neutral', 'male', 'female'],
                        help='Use gender neutral or gender specific SMPL' +
                        'model')
    parser.add_argument('--float_dtype', type=str, default='float32',
                        help='The types of floats used')
    parser.add_argument('--model_type', default='smpl', type=str,
                        choices=['smpl', 'smplh', 'smplx'],
                        help='The type of the model that we will fit to the' +
                        ' data.')
    parser.add_argument('--camera_type', type=str, default='persp',
                        choices=['persp'],
                        help='The type of camera used')
    parser.add_argument('--optim_jaw', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Optimize over the jaw pose')
    parser.add_argument('--optim_hands', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Optimize over the hand pose')
    parser.add_argument('--optim_expression', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Optimize over the expression')
    parser.add_argument('--optim_shape', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Optimize over the shape space')

    parser.add_argument('--model_folder',
                        default='models',
                        type=str,
                        help='The directory where the models are stored.')
    parser.add_argument('--use_joints_conf', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the confidence scores for the optimization')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='The size of the batch')
    parser.add_argument('--num_gaussians',
                        default=8,
                        type=int,
                        help='The number of gaussian for the Pose Mixture' +
                        ' Prior.')
    parser.add_argument('--use_pca', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the low dimensional PCA space for the hands')
    parser.add_argument('--num_pca_comps', default=6, type=int,
                        help='The number of PCA components for the hand.')
    parser.add_argument('--flat_hand_mean', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use the flat hand as the mean pose')
    parser.add_argument('--body_prior_type', default='mog', type=str,
                        help='The type of prior that will be used to' +
                        ' regularize the optimization. Can be a Mixture of' +
                        ' Gaussians (mog)')
    parser.add_argument('--left_hand_prior_type', default='mog', type=str,
                        choices=['mog', 'l2', 'None'],
                        help='The type of prior that will be used to' +
                        ' regularize the optimization of the pose of the' +
                        ' left hand. Can be a Mixture of' +
                        ' Gaussians (mog)')
    parser.add_argument('--right_hand_prior_type', default='mog', type=str,
                        choices=['mog', 'l2', 'None'],
                        help='The type of prior that will be used to' +
                        ' regularize the optimization of the pose of the' +
                        ' right hand. Can be a Mixture of' +
                        ' Gaussians (mog)')
    parser.add_argument('--jaw_prior_type', default='l2', type=str,
                        choices=['l2', 'None'],
                        help='The type of prior that will be used to' +
                        ' regularize the optimization of the pose of the' +
                        ' jaw.')
    parser.add_argument('--use_vposer', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use the VAE pose embedding')
    parser.add_argument('--vposer_ckpt', type=str, default='',
                        help='The path to the V-Poser checkpoint')
    # Left/Right shoulder and hips
    parser.add_argument('--init_joints_idxs', nargs='*', type=int,
                        default=[9, 12, 2, 5],
                        help='Which joints to use for initializing the camera')
    parser.add_argument('--body_tri_idxs', nargs='*',
                        #  default='5.12,2.9',
                        default=[5, 12, 2, 9],
                        type=int,
                        help='The indices of the joints used to estimate' +
                        ' the initial depth of the camera. The format' +
                        ' should be vIdx1 vIdx2 vIdx3 vIdx4')

    parser.add_argument('--prior_folder', type=str, default='prior',
                        help='The folder where the prior is stored')
    parser.add_argument('--focal_length',
                        default=5000,
                        type=float,
                        help='Value of focal length.')
    parser.add_argument('--rho',
                        default=100,
                        type=float,
                        help='Value of constant of robust loss')
    parser.add_argument('--interpenetration',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to use the interpenetration term')
    parser.add_argument('--penalize_outside',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Penalize outside')
    parser.add_argument('--data_weights', nargs='*',
                        default=[1, ] * 5, type=float,
                        help='The weight of the data term')
    parser.add_argument('--body_pose_prior_weights',
                        default=[4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
                        nargs='*',
                        type=float,
                        help='The weights of the body pose regularizer')
    parser.add_argument('--shape_weights',
                        default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
                        type=float, nargs='*',
                        help='The weights of the Shape regularizer')
    parser.add_argument('--expr_weights',
                        default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
                        type=float, nargs='*',
                        help='The weights of the Expressions regularizer')
    parser.add_argument('--face_joints_weights',
                        default=[0.0, 0.0, 0.0, 2.0], type=float,
                        nargs='*',
                        help='The weights for the facial keypoints' +
                        ' for each stage of the optimization')
    parser.add_argument('--hand_joints_weights',
                        default=[0.0, 0.0, 0.0, 2.0],
                        type=float, nargs='*',
                        help='The weights for the 2D joint error of the hands')
    parser.add_argument('--jaw_pose_prior_weights',
                        nargs='*',
                        help='The weights of the pose regularizer of the' +
                        ' hands')
    parser.add_argument('--hand_pose_prior_weights',
                        default=[1e2, 5 * 1e1, 1e1, .5 * 1e1],
                        type=float, nargs='*',
                        help='The weights of the pose regularizer of the' +
                        ' hands')
    parser.add_argument('--coll_loss_weights',
                        default=[0.0, 0.0, 0.0, 2.0], type=float,
                        nargs='*',
                        help='The weight for the collision term')

    parser.add_argument('--depth_loss_weight', default=1e2, type=float,
                        help='The weight for the regularizer for the' +
                        ' z coordinate of the camera translation')
    parser.add_argument('--df_cone_height', default=0.5, type=float,
                        help='The default value for the height of the cone' +
                        ' that is used to calculate the penetration distance' +
                        ' field')
    parser.add_argument('--max_collisions', default=8, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--point2plane', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use point to plane distance')
    parser.add_argument('--part_segm_fn', default='', type=str,
                        help='The file with the part segmentation for the' +
                        ' faces of the model')
    parser.add_argument('--ign_part_pairs', default=None,
                        nargs='*', type=str,
                        help='Pairs of parts whose collisions will be ignored')
    parser.add_argument('--use_hands', default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the hand keypoints in the SMPL' +
                        'optimization process')
    parser.add_argument('--use_face', default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the facial keypoints in the optimization' +
                        ' process')
    parser.add_argument('--use_face_contour', default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the dynamic contours of the face')
    parser.add_argument('--side_view_thsh',
                        default=25,
                        type=float,
                        help='This is thresholding value that determines' +
                        ' whether the human is captured in a side view.' +
                        'If the pixel distance between the shoulders is less' +
                        ' than this value, two initializations of SMPL fits' +
                        ' are tried.')
    parser.add_argument('--optim_type', type=str, default='adam',
                        help='The optimizer used')
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='The learning rate for the algorithm')
    parser.add_argument('--gtol', type=float, default=1e-8,
                        help='The tolerance threshold for the gradient')
    parser.add_argument('--ftol', type=float, default=2e-9,
                        help='The tolerance threshold for the function')
    parser.add_argument('--maxiters', type=int, default=100,
                        help='The maximum iterations for the optimization')

    args = parser.parse_args(argv)

    args_dict = vars(args)

    assert len(args_dict['body_tri_idxs']) % 2 == 0, (
        'Number of body_tri_idxs arguments must be divisble by 2.'
        f' Got: {len(args_dict["body_tri_idxs"])}'
    )
    num_tri_idxs = len(args_dict['body_tri_idxs'])
    # Convert the list of indices to a list of pairs
    args_dict['body_tri_idxs'] = [
        (args_dict['body_tri_idxs'][ii], args_dict['body_tri_idxs'][jj])
        for ii, jj in zip(range(0, num_tri_idxs, 2), range(1, num_tri_idxs, 2))
    ]
    return args_dict
