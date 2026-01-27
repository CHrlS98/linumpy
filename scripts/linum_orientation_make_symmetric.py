#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import argparse
import nibabel as nib

from dipy.reconst.shm import sh_to_sf_matrix, order_from_ncoef
from dipy.core.sphere import Sphere
from dipy.data import get_sphere


SH_BASES = {
    'descoteaux07_legacy': ('descoteaux07', True),
    'tournier07_legacy': ('tournier07', True),
    'descoteaux07': ('descoteaux07', False),
    'tournier07': ('tournier07', False)
}


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh')
    p.add_argument('out_sh')
    p.add_argument('--index', type=int)
    p.add_argument('--sh_basis', choices=SH_BASES.keys(), default='tournier07',
                   help='SH basis for hist-FOD. [%(default)s]')
    return p


def get_rotation_matrix(sh_order_max, basis_type, legacy):
    sphere_a = get_sphere(name='repulsion100')
    sphere_b = Sphere(x=-sphere_a.vertices[:, 0],
                      y= sphere_a.vertices[:, 1],
                      z= sphere_a.vertices[:, 2])
    print(sphere_b.vertices.shape)
    sh_to_sf_mat = sh_to_sf_matrix(sphere_a,
                                   basis_type=basis_type,
                                   legacy=legacy,
                                   sh_order_max=sh_order_max,
                                   return_inv=False)
    _, sf_to_sh_mat = sh_to_sf_matrix(sphere_b,
                                      basis_type=basis_type,
                                      legacy=legacy,
                                      sh_order_max=sh_order_max,
                                      return_inv=True)
    T_rot = sh_to_sf_mat.dot(sf_to_sh_mat)
    return T_rot


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    sh_im = nib.load(args.in_sh)
    sh = sh_im.get_fdata()

    basis_type, legacy = SH_BASES[args.sh_basis]
    order = order_from_ncoef(sh.shape[-1])
    T_r = get_rotation_matrix(order, basis_type, legacy)

    sh_flip = sh.dot(T_r)
    sh_flip = sh_flip[::-1]

    out_sh = sh_flip / 2.0 + sh / 2.0

    nib.save(nib.Nifti1Image(out_sh.astype(sh_im.get_data_dtype()), sh_im.affine), args.out_sh)


if __name__ == '__main__':
    main()
