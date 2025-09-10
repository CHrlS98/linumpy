#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Workflow Description
// Creates a 3D volume from raw S-OCT tiles
// Input: Directory containing raw data set tiles
// Output: 3D reconstruction

// Parameters
params.input = ""
params.shifts_xy = "${params.input}/shifts_xy.csv"
params.output = "output"
params.resolution = 10 // Resolution of the reconstruction in micron/pixel
params.processes = 1 // Maximum number of python processes per nextflow process
params.axial_resolution = 1.5 // Axial resolution of imaging system in microns
params.crop_interface_out_depth = 600 // Minimum depth of the cropped image in microns
params.use_old_folder_structure = false // Use the old folder structure where tiles are not stored in subfolders based on their Z
params.method = "affine" // Method for stitching, can be 'euler' or 'affine'
params.grad_mag_tolerance = 1e-12 // Gradient magnitude tolerance for the 3D stacking algorithm
params.metric = "MI" // Metric for the 3D stacking algorithm

process resample_mosaic_grid {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("mosaic_grid_3d_${params.resolution}um.ome.zarr")
    script:
    """
    tar -xvf $mosaic_grid --directory out_untar
    mv out_untar/*.ome.zarr ./mosaic_grid_z${slice_id}.ome.zarr
    linum_resample_mosaic_grid.py mosaic_grid_z${slice_id}.ome.zarr "mosaic_grid_3d_${params.resolution}um.ome.zarr" -r ${params.resolution}

    # cleanup; we don't need these temp files in our working directory
    rm -rf mosaic_grid_z${slice_id}.ome.zarr
    """
}

process clip_outliers {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("mosaic_grid_3d_${params.resolution}um_clipped.ome.zarr")
    script:
    """
    linum_clip_percentile.py $mosaic_grid mosaic_grid_3d_${params.resolution}um_clipped.ome.zarr
    """
}

process fix_focal_curvature {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("*_focalFix.ome.zarr")
    script:
    """
    linum_detect_focal_curvature.py ${mosaic_grid} mosaic_grid_3d_${params.resolution}um_focalFix.ome.zarr
    """
}

process fix_illumination {
    cpus params.processes

    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("*_illuminationFix.ome.zarr")
    script:
    """
    linum_fix_illumination_3d.py ${mosaic_grid} mosaic_grid_3d_${params.resolution}um_illuminationFix.ome.zarr --n_processes ${params.processes}
    """
}

process generate_aip {
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("aip.ome.zarr")
    script:
    """
    linum_aip.py ${mosaic_grid} aip.ome.zarr
    """
}

process estimate_xy_transformation {
    input:
        tuple val(slice_id), path(aip)
    output:
        tuple val(slice_id), path("transform_xy.npy")
    script:
    """
    linum_estimate_transform.py ${aip} transform_xy.npy
    """
}

process stitch_3d {
    input:
        tuple val(slice_id), path(mosaic_grid), path(transform_xy)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_${params.resolution}um.ome.zarr")
    script:
    """
    linum_stitch_3d.py ${mosaic_grid} ${transform_xy} slice_z${slice_id}_${params.resolution}um.ome.zarr
    """
}

process beam_profile_correction {
    input:
        tuple val(slice_id), path(slice_3d)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_${params.resolution}um_axial_corr.ome.zarr")
    script:
    """
    linum_compensate_psf_model_free.py ${slice_3d} "slice_z${slice_id}_${params.resolution}um_axial_corr.ome.zarr"
    """
}

process attenuation_correction {
    input:
        tuple val(slice_id), path(slice_3d)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_${params.resolution}um_attn_corr.ome.zarr")
    script:
    """
    linum_compute_attenuation.py ${slice_3d} "slice_z${slice_id}_${params.resolution}um_attn.ome.zarr" --mask_all
    linum_compute_attenuation_bias_field.py "slice_z${slice_id}_${params.resolution}um_attn.ome.zarr" "slice_z${slice_id}_${params.resolution}um_attn_bias.ome.zarr" --isInCM
    linum_compensate_attenuation.py ${slice_3d} "slice_z${slice_id}_${params.resolution}um_attn_bias.ome.zarr" "slice_z${slice_id}_${params.resolution}um_attn_corr.ome.zarr"
    """
}

process crop_interface {
    input:
        tuple val(slice_id), path(image)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_${params.resolution}um_crop.ome.zarr")
    script:
    """
    linum_crop_3d_mosaic_below_interface.py $image "slice_z${slice_id}_${params.resolution}um_crop.ome.zarr" --depth $params.crop_interface_out_depth --crop_before_interface --pad_after
    """
}

process stack_mosaics_into_3d_volume {
    publishDir "$params.output/$task.process"
    input:
        tuple path("inputs/*"), path("shifts_xy.csv")
    output:
        tuple path("3d_stack_mosaic_${params.resolution}um.ome.zarr"), path("3d_stack_mosaic_${params.resolution}um_offsets.npy")
    script:
    """
    linum_stack_mosaics_into_3d_volume_v2.py inputs shifts_xy.csv 3d_stack_mosaic_${params.resolution}um.ome.zarr 3d_stack_mosaic_${params.resolution}um_offsets.npy
    """
}

process estimate_pairwise_transform {
    input:
        tuple val(slice_id), path(volume), path(offsets), val(min_ind)
    output:
        tuple val(slice_id), path("slice_z${slice_id}_transform.mat"), path("slice_z${slice_id}_pairwise_transform.png")
    script:
    """
    linum_estimate_transform_pairwise.py ${volume} ${offsets} ${slice_id} slice_z${slice_id}_transform.mat --first_slice_index ${min_ind} --method ${params.method} --metric ${params.metric} --screenshot slice_z${slice_id}_pairwise_transform.png
    """
}

process apply_transforms_to_stack {
    publishDir "$params.output/$task.process"
    input:
        tuple path("inputs/*"), path(volume), path(offsets), val(min_ind)
    output:
        path("3d_stack_${params.resolution}um_aligned.ome.zarr")
    script:
    """
    linum_apply_transforms_to_stack.py ${volume} ${offsets} inputs 3d_stack_${params.resolution}um_aligned.ome.zarr --first_slice_index ${min_ind}
    """
}

workflow {
    inputSlices = Channel.fromFilePairs("$params.input/*.ome.zarr.zip")
    inputSlices.view()

    shifts_xy = Channel.fromPath("$params.shifts_xy")
    shifts_xy.view()

    // Generate a 3D mosaic grid.
    resample_mosaic_grid(inputSlices)

    // Clip values to remove hyperintensities
    clip_outliers(resample_mosaic_grid.out)

    // Focal plane curvature compensation
    fix_focal_curvature(clip_outliers.out)

    // Compensate for XY illumination inhomogeneity
    fix_illumination(fix_focal_curvature.out)

    // Generate AIP mosaic grid
    generate_aip(fix_illumination.out)

    // Extract tile position (XY) from AIP mosaic grid
    estimate_xy_transformation(generate_aip.out)

    // Stitch the tiles in 3D mosaics
    stitch_3d(fix_illumination.out.combine(estimate_xy_transformation.out, by:0))

    // Crop at interface
    crop_interface(stitch_3d.out)

    // PSF correction
    beam_profile_correction(crop_interface.out)

    // Attenuation correction
    attenuation_correction(beam_profile_correction.out)

    // 3D mosaic
    stack_in_channel = attenuation_correction.out
        .toSortedList{a, b -> a[0] <=> b[0]}
        .flatten().collate(2)
        .map{_meta, filename -> filename}.collect()
        .merge(shifts_xy){a, b -> tuple(a, b)}
    stack_mosaics_into_3d_volume(stack_in_channel)

    all_indices = inputSlices.map{meta, _files -> meta}
    min_indice = all_indices.min().toInteger()

    // Collect slice IDs for the pairwise transforms
    estimate_pairwise_channel = all_indices
        .filter{v -> v as Integer != min_indice.val}
        .combine(stack_mosaics_into_3d_volume.out)
        .combine(min_indice)

    // Estimate pairwise transforms
    estimate_pairwise_transform(estimate_pairwise_channel)

    // Use transforms to align the 3D volume
    apply_transforms_to_stack_channel = estimate_pairwise_transform.out
        .map{_meta, slice_3d, _screenshot -> slice_3d}
        .collect().map{it -> [it]}
        .combine(stack_mosaics_into_3d_volume.out)
        .combine(min_indice)

    apply_transforms_to_stack(apply_transforms_to_stack_channel)
}
