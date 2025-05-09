#!/usr/bin/env python3
"""
Rigid registration of multiple subjects using landmarks. Landmarks are placed
between both brain hemispheres and described below:
    * AC: Center of the anterior commissure
    * CC_A: Anterior portion of the bridge of the corpus callosum
    * CC_P: Posterior portion of the bridge of the corpus callosum

The script first translates each brain to the average position of the anterior
commissure. Then, an average interhemispheric plane is computed and all brains
are aligned to this reference. Finally, the planes are rotated around the AC
anchor such that the pairwise distance between all CC_A and the pairwise
distance between CC_P are minimized.

The script expects nifti images and stores the resulting transform in the image
header.
"""
import argparse
import numpy as np
import nibabel as nib
import json
import os

from scipy.spatial.transform import Rotation
from scipy.optimize import minimize_scalar


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_json',
                   help='Input json file containing registration parameters.')
    p.add_argument('in_root',
                   help='Input paths in in_json are given relative to this path.')
    p.add_argument('out_dir',
                   help='Output paths are given relative to this path.')
    return p


def get_mean_ac_anchor_voxum(register_params):
    anchors_ac = np.asarray([item.ac_world_homogeneous for item in register_params])
    mean_anchor_ac_voxum = np.mean(anchors_ac, axis=0)

    return mean_anchor_ac_voxum


class ImageRegistrationItem:
    def __init__(self, in_file, out_file, ac_anchor,
                 cc_a_anchor, cc_p_anchor):
        self._image = nib.load(in_file)
        self._affine_so_far = self._image.affine
        self._ac_vox_homogeneous = np.asarray(np.append(ac_anchor, [1.])).reshape((4, 1))
        self._cc_a_vox_homogeneous = np.asarray(np.append(cc_a_anchor, [1.])).reshape((4, 1))
        self._cc_p_vox_homogeneous = np.asarray(np.append(cc_p_anchor, [1.])).reshape((4, 1))
        self._out_filename = out_file

    @property
    def output_filename(self):
        return self._out_filename

    @property
    def image(self):
        return self._image

    @property
    def ac_world_homogeneous(self):
        return self._affine_so_far @ self._ac_vox_homogeneous

    @property
    def cc_a_world_homogeneous(self):
        return self._affine_so_far @ self._cc_a_vox_homogeneous

    @property
    def cc_p_world_homogeneous(self):
        return self._affine_so_far @ self._cc_p_vox_homogeneous

    @property
    def affine_so_far(self):
        return self._affine_so_far

    @affine_so_far.setter
    def affine_so_far(self, affine_so_far):
        self._affine_so_far = affine_so_far


def format_register_params(register_params, root_dir, out_dir):
    image_registration_items = []
    for image_dict in register_params:
        infile = os.path.join(root_dir, image_dict['in_file'])
        outfile = os.path.join(out_dir, image_dict['out_file'])
        ac_anchor = image_dict['AC']
        cc_a_anchor = image_dict['CC_A']
        cc_p_anchor = image_dict['CC_P']
        curr_imreg_item = ImageRegistrationItem(infile, outfile, ac_anchor,
                                                cc_a_anchor, cc_p_anchor)
        image_registration_items.append(curr_imreg_item)

    return image_registration_items


def objective_anchors_align(theta, origin_cc_a, origin_cc_p,
                            target_cc_a, target_cc_p):
    """
    
    Parameters
    ----------
    theta: float
        Parameter to optimize.
    origin_cc_a: ndarray (3,)
        3D position of cc_a anchor.
    origin_cc_p: ndarray (3,)
        3D position of cc_p anchor.
    target_cc_a: ndarray (3,)
        Target position of cc_a anchor.
    target_cc_p: ndarray (3,)
        Target position of cc_p anchor.
    """
    return 0.5 * (
        (target_cc_a[0] - origin_cc_a[0])**2 +
        (target_cc_a[1] - origin_cc_a[1] * np.cos(theta) + origin_cc_a[2] * np.sin(theta))**2 +
        (target_cc_a[2] - origin_cc_a[1] * np.sin(theta) - origin_cc_a[2] * np.cos(theta))**2 +
        (target_cc_p[0] - origin_cc_p[0])**2 +
        (target_cc_p[1] - origin_cc_p[1] * np.cos(theta) + origin_cc_p[2] * np.sin(theta))**2 +
        (target_cc_p[2] - origin_cc_p[1] * np.sin(theta) - origin_cc_p[2] * np.cos(theta))**2
    )


def jacobian_anchors_align(theta, origin_cc_a, origin_cc_p,
                           target_cc_a, target_cc_p):
    """
    Parameters
    ----------
    theta: float
        Parameter to optimize.
    origin_cc_a: ndarray (3,)
        3D position of cc_a anchor.
    origin_cc_p: ndarray (3,)
        3D position of cc_p anchor.
    target_cc_a: ndarray (3,)
        Target position of cc_a anchor.
    target_cc_p: ndarray (3,)
        Target position of cc_p anchor.
    """
    return (
        (target_cc_a[1] - origin_cc_a[1]*np.cos(theta) + origin_cc_a[2]*np.sin(theta))*
        (origin_cc_a[1]*np.sin(theta) + origin_cc_a[2]*np.cos(theta)) +
        (target_cc_a[2] - origin_cc_a[1]*np.sin(theta) - origin_cc_a[2]*np.cos(theta))*
        (-origin_cc_a[1]*np.cos(theta) + origin_cc_a[2]*np.sin(theta)) +
        (target_cc_p[1] - origin_cc_p[1]*np.cos(theta) + origin_cc_p[2]*np.sin(theta))*
        (origin_cc_p[1]*np.sin(theta) + origin_cc_p[2]*np.cos(theta)) +
        (target_cc_p[2] - origin_cc_p[1]*np.sin(theta) - origin_cc_p[2]*np.cos(theta))*
        (-origin_cc_p[1]*np.cos(theta) + origin_cc_p[2]*np.sin(theta))
    )


def rotation_around_origin(rotation, origin_homogeneous):
    translation_matrix = np.identity(4)
    translation_matrix[:3, 3] = -origin_homogeneous[:3].reshape((-1,))
    translation_matrix_inv = np.identity(4)
    translation_matrix_inv[:3, 3] = origin_homogeneous[:3].reshape((-1,))
    rotation_matrix = translation_matrix_inv @ (rotation @ translation_matrix)
    return rotation_matrix


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    with open(args.in_json) as f:
        register_params = json.load(f)['params']

    image_registration_items = format_register_params(register_params,
                                                      args.in_root,
                                                      args.out_dir)

    # compute mean AC anchor position in voxum space
    mean_ac_voxum = get_mean_ac_anchor_voxum(image_registration_items)

    # we want each subject's affine to bring anchor_ac to mean_anchor_ac_voxmm
    # this means we must apply a translation to affine@anchors
    for imreg_item in image_registration_items:
        translation_matrix = np.identity(4)
        translation = mean_ac_voxum - imreg_item.ac_world_homogeneous
        translation_matrix[:3, 3] = translation[:3, 0]
        imreg_item.affine_so_far = translation_matrix @ imreg_item.affine_so_far

    # next step is rotating the brain such than that triangle
    # defined by the anchors lies on the interhemispheric plane
    target_normal = np.array([[1.0], [0.0], [0.0]])
    for imreg_item in image_registration_items:
        ac_cc_a = imreg_item.cc_a_world_homogeneous - imreg_item.ac_world_homogeneous
        ac_cc_p = imreg_item.cc_p_world_homogeneous - imreg_item.ac_world_homogeneous
        curr_normal = np.cross(ac_cc_a[:3], ac_cc_p[:3], axis=0)
        curr_normal /= np.linalg.norm(curr_normal)
        theta = np.arccos(curr_normal.T.dot(target_normal))
        rotvec = np.cross(curr_normal, target_normal, axis=0)
        rotvec_norm = np.linalg.norm(rotvec)
        if rotvec_norm > 0.0:
            rotvec /= rotvec_norm
            r = Rotation.from_rotvec(theta*rotvec.reshape((-1,)))
            rotation_matrix = np.identity(4)
            rotation_matrix[:3, :3] = r.as_matrix()
            rotation_matrix = rotation_around_origin(rotation_matrix, imreg_item.ac_world_homogeneous)
            imreg_item.affine_so_far = rotation_matrix @ imreg_item.affine_so_far

    # rotate around normal s.t. the distance between anchor points is minimized
    target_cc_a = []
    target_cc_p = []
    for imreg_item in image_registration_items:
        origin_cc_a = imreg_item.cc_a_world_homogeneous[:3]
        origin_cc_p = imreg_item.cc_p_world_homogeneous[:3]
        target_cc_a.append(origin_cc_a)
        target_cc_p.append(origin_cc_p)
    target_cc_a = np.mean(target_cc_a, axis=0) - mean_ac_voxum[:3]
    target_cc_p = np.mean(target_cc_p, axis=0) - mean_ac_voxum[:3]

    # Rotate around AC
    for imreg_item in image_registration_items:
        origin_cc_a = imreg_item.cc_a_world_homogeneous[:3]
        origin_cc_p = imreg_item.cc_p_world_homogeneous[:3]
        origin_cc_a -= imreg_item.ac_world_homogeneous[:3]
        origin_cc_p -= imreg_item.ac_world_homogeneous[:3]
        print(origin_cc_a, origin_cc_p, target_cc_a, target_cc_p)

        res = minimize_scalar(lambda x: objective_anchors_align(x, origin_cc_a, origin_cc_p,
                                                                target_cc_a, target_cc_p),
                              bounds=(-np.pi, np.pi))
        theta = res.x[0]
        print(theta)
        rotation_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, np.cos(theta), -np.sin(theta), 0.0],
                                    [0.0, np.sin(theta), np.cos(theta), 0.0],
                                    [0.0, 0.0, 0.0, 1.0]])
        rotation_matrix = rotation_around_origin(rotation_matrix, mean_ac_voxum)
        imreg_item.affine_so_far = rotation_matrix @ imreg_item.affine_so_far

    # save the resulting affine transformation
    for imreg_item in image_registration_items:
        # print(imreg_item.affine_so_far)
        outdirs, _ = os.path.split(imreg_item.output_filename)
        if not os.path.exists(outdirs):
            os.makedirs(outdirs)
        nib.save(nib.Nifti1Image(imreg_item.image.get_fdata(), imreg_item.affine_so_far),
                 imreg_item.output_filename)


if __name__ == '__main__':
    main()
