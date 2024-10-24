import shutil
from pathlib import Path
from typing import List, Tuple, Any, Callable, Union
from itertools import product
from importlib.metadata import version

import dask.array as da
import numpy as np
import zarr
from numpy import ndarray
from ome_zarr.dask_utils import resize as dask_resize
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_multiscale, write_image
from skimage.transform import resize, pyramid_gaussian

""" 
    This file contains functions for working with zarr files
"""

def create_transformation_dict(nlevels, ndims=3):
    """
    Create a dictionary with the transformation information for 3D images.

    :param scales: The scale of the image, in z y x order.
    :param levels: The number of levels in the pyramid.
    :return:
    """
    def _get_scale(level, ndims):
        scale_def = [1.0, (2**level), (2**level), (2**level)]
        offset = len(scale_def) - ndims
        return scale_def[offset:]

    coord_transforms = []
    for i in range(nlevels):
        transform_dict = [{
            "type": "scale",
            "scale": _get_scale(i, ndims)
        }]
        coord_transforms.append(transform_dict)
    return coord_transforms


def generate_axes_dict(ndims=3):
    """
    Generate the axes dictionary for the zarr file.

    :return: The axes dictionary
    """
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "millimeter"},
        {"name": "y", "type": "space", "unit": "millimeter"},
        {"name": "x", "type": "space", "unit": "millimeter"}
    ]
    offset = len(axes) - ndims
    return axes[offset:]


def create_directory(store_path, overwrite=False):
    directory = Path(store_path)
    if directory.exists():
        if overwrite:
            shutil.rmtree(directory)
        else:
            raise FileExistsError('Directory {} already exists. '
                                  'Set overwrite=True to overwrite.'
                                  .format(directory.as_posix()))
    directory.mkdir(parents=True)
    return directory


def save_zarr(data, store_path, scale, *,
              chunks=(128, 128, 128), n_levels=5,
              overwrite=False):
    """
    Save numpy array to disk in zarr format following OME-NGFF file specifications.
    Expected ordering for axes in `data` and `scale` is `(c, z, y, x)`.

    :type data: numpy or dask array
    :param data: numpy or dask array to save as zarr.
    :type store_path: str
    :param store_path: The path of the output zarr group.
    :type scale: tuple of n `float`, with n the number of dimensions.
    :param scale: Voxel size in mm.
    :type chunks: tuple of n `int`, with n the number of dimensions.
    :param chunks: Chunk size on disk.
    :type n_levels: int
    :param n_levels: Number of levels in Gaussian pyramid.
    :type overwrite: bool
    :param overwrite: Overwrite `store_path` if it already exists.

    :return  zarr_group: Resulting zarr group saved to disk.
    :type zarr_group: zarr.hierarchy.group
    """
    # pyramidal decomposition (ome_zarr.scale.Scaler) keywords
    pyramid_kw = {"max_layer": n_levels,
                  "method": "nearest",
                  "downscale": 2}
    ndims = len(data.shape)

    # pyramid = list(pyramid_gaussian(data, **pyramid_kw))
    # metadata describes the downsampling method used for generating
    # multiscale data representation (see also type in write_multiscale)
    metadata = {"method": "ome_zarr.scale.Scaler",
                "version":version("ome-zarr"),
                "args": pyramid_kw}

    # axes and coordinate transformations
    axes = generate_axes_dict(ndims)
    coordinate_transformations = create_transformation_dict(n_levels+1, ndims)

    # create directory for zarr storage
    create_directory(store_path, overwrite)
    
    store = parse_url(store_path, mode='w').store
    zarr_group = zarr.group(store=store)

    # the base transformation is applied to all levels of the pyramid
    # and describes the original voxel size of the dataset
    base_coord_transformation = [
        {"type":"scale", "scale": list(scale)}
    ]
    write_image(data, zarr_group, storage_options=dict(chunks=chunks),
                scaler=Scaler(**pyramid_kw),
                axes=axes, coordinate_transformations=coordinate_transformations,
                compute=True, metadata=metadata, type="gaussian",
                coordinateTransformations=base_coord_transformation)

    # return zarr group containing saved data
    return zarr_group


def temp_store_nifti_to_zarr(image, chunks, dtype=np.float32):
    store = zarr.TempStore(dir=".")
    zarr_group = zarr.open(store, mode='w')
    zarr_array = zarr_group.zeros('0', shape=image.shape, chunks=chunks, dtype=dtype)

    blocks_dims = (int(float(image.shape[0]) / chunks[0] + 0.5),
                   int(float(image.shape[1]) / chunks[1] + 0.5),
                   int(float(image.shape[2]) / chunks[2] + 0.5))
    print(blocks_dims, chunks)
    for (i, j, k) in product(*[range(_i) for _i in blocks_dims]):
        print(i, j, k)
        img_block = image.slicer[i*chunks[0]:min((i+1)*chunks[0], image.shape[0]),
                                 j*chunks[1]:min((j+1)*chunks[1], image.shape[1]),
                                 k*chunks[2]:min((k+1)*chunks[2], image.shape[2]),
                                 ...]
        print(img_block.shape)
        data = img_block.get_fdata(dtype=dtype)
        print('loaded nifti')
        zarr_array[i*chunks[0]:min((i+1)*chunks[0], image.shape[0]),
                   j*chunks[1]:min((j+1)*chunks[1], image.shape[1]),
                   k*chunks[2]:min((k+1)*chunks[2], image.shape[2])] = data

    return zarr_array
