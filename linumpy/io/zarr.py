import shutil
from pathlib import Path
from importlib.metadata import version

import dask.array as da
import zarr
from ome_zarr.dask_utils import resize as dask_resize
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from ome_zarr.reader import Reader, Multiscales
from skimage.transform import resize

""" 
    This file contains functions for working with zarr files
"""


class CustomScaler(Scaler):
    def resize_image(self, image):
        """
        Resize a numpy array OR a dask array to a smaller array (not pyramid)
        """
        if isinstance(image, da.Array):

            def _resize(image, out_shape, **kwargs):
                return dask_resize(image, out_shape, **kwargs)

        else:
            _resize = resize

        # downsample in X, Y, and Z.
        new_shape = list(image.shape)
        new_shape[-1] = image.shape[-1] // self.downscale
        new_shape[-2] = image.shape[-2] // self.downscale
        new_shape[-3] = image.shape[-3] // self.downscale
        out_shape = tuple(new_shape)

        dtype = image.dtype
        image = _resize(
            image.astype(float), out_shape, order=1, mode="reflect", anti_aliasing=False
        )
        return image.astype(dtype)

    def _by_plane(self, base, func):
        # This method is called by base class when interpolation methods (e.g. nearest)
        # directly. Because `write_image` never call these methods, we don't need to
        # implement it here. We raise an error to make sure the CustomScaler class is not
        # used for this purpose.
        raise NotImplementedError("_by_plane method not implemented for CustomScaler")


def create_transformation_dict(nlevels, ndims=3):
    """
    Create a dictionary with the transformation information for 3D images.

    :param scales: The scale of the image, in z y x order.
    :param levels: The number of levels in the pyramid.
    :return:
    """
    def _get_scale(level, ndims):
        scale_def = [1.0, (2.0**level), (2.0**level), (2.0**level)]
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
                  "method": "nearest",  # cannot be a value other than `nearest`
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
                scaler=CustomScaler(**pyramid_kw),
                axes=axes, coordinate_transformations=coordinate_transformations,
                compute=True, metadata=metadata, type="gaussian",
                coordinateTransformations=base_coord_transformation)

    # return zarr group containing saved data
    return zarr_group


def read_multiscale(zarr_path, mode='r'):
    """
    
    """
    root_location = parse_url(zarr_path, mode=mode)
    r = Reader(root_location)
    for node in r():
        if Multiscales.matches(node.zarr):
            return Multiscales(node)
    raise ValueError(f'{zarr_path} does not contain multiscale data.')
