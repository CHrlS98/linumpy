import argparse
import nibabel as nib
import numpy as np

def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--in_volumes', nargs='+', required=True,
                   help='Registered nifti images.')
    p.add_argument('--out_volume', required=True)
    p.add_argument('--mode', choices=['mean', 'max'], default='mean')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    ref_im = nib.load(args.in_volumes[0])
    out_data = np.zeros((ref_im.shape))
    for v in args.in_volumes:
        im = nib.load(v)
        data = im.get_fdata()
        if args.mode == 'mean':
            out_data += data
        else:
            out_data = np.maximum(out_data, data)

    if args.mode == 'mean':
        out_data = out_data / len(args.in_volumes)
    nib.save(nib.Nifti1Image(out_data, ref_im.affine), args.out_volume)


if __name__ == '__main__':
    main()
