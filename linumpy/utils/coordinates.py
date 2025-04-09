# -*- coding:utf-8 -*-

AXIS_NAME_TO_INDEX = {
    'sagittal': 0,
    'coronal': 1,
    'axial': 2
}

def slice_along_axis(idx, shape, axis=-1):
    slicer = [slice(shape[i]) for i in range(len(shape))]
    slicer[axis] = idx
    return tuple(slicer)
