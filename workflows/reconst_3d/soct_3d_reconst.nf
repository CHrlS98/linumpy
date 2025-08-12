#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Workflow Description
// Creates a 3D volume from raw S-OCT tiles
// Input: Directory containing raw data set tiles
// Output: 3D reconstruction

// Parameters
params.input = ""
params.output = "output"
params.resolution = 10 // Resolution of the reconstruction in micron/pixel
params.processes = 1 // Maximum number of python processes per nextflow process
params.axial_resolution = 1.5 // Axial resolution of imaging system in microns
params.crop_interface_out_depth = 600 // Minimum depth of the cropped image in microns
params.use_old_folder_structure = false // Use the old folder structure where tiles are not stored in subfolders based on their Z
params.method = "affine" // Method for stitching, can be 'euler' or 'affine'
params.grad_mag_tolerance = 1e-12 // Gradient magnitude tolerance for the 3D stacking algorithm
params.metric = "MI" // Metric for the 3D stacking algorithm

// Processes
process create_mosaic_grid {
    input:
        tuple val(slice_id), path(tiles)
    output:
        tuple val(slice_id), path("*.ome.zarr")
    script:
    """
    linum_create_mosaic_grid_3d.py mosaic_grid_3d_${params.resolution}um.ome.zarr --from_tiles_list $tiles --resolution ${params.resolution} --n_processes ${params.processes} --axial_resolution ${params.axial_resolution}
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
    linum_compensate_attenuation.py "slice_z${slice_id}_${params.resolution}um_attn.ome.zarr" "slice_z${slice_id}_${params.resolution}um_attn_bias.ome.zarr" "slice_z${slice_id}_${params.resolution}um_attn_corr.ome.zarr"
    """
}

process estimate_xy_shifts_from_metadata {
    publishDir "$params.output/$task.process"
    input:
        path(input_dir)
    output:
        path("shifts_xy.csv")
    script:
    """
    linum_estimate_xy_shift_from_metadata.py ${input_dir} shifts_xy.csv
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

workflow {
    if (params.use_old_folder_structure)
    {
        inputSlices = Channel.fromPath("$params.input/tile_x*_y*_z*/", type: 'dir')
                            .map{path -> tuple(path.toString().substring(path.toString().length() - 2), path)}
                            .groupTuple()
    }
    else
    {
        inputSlices = Channel.fromPath("$params.input/**/tile_x*_y*_z*/", type: 'dir')
                            .map{path -> tuple(path.toString().substring(path.toString().length() - 2), path)}
                            .groupTuple()
    }
    input_dir_channel = Channel.fromPath("$params.input", type: 'dir')

    // Generate a 3D mosaic grid.
    create_mosaic_grid(inputSlices)

    // Estimate XY shifts from metadata
    estimate_xy_shifts_from_metadata(input_dir_channel)

    // Focal plane curvature compensation
    fix_focal_curvature(create_mosaic_grid.out)

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
        .merge(estimate_xy_shifts_from_metadata.out){a, b -> tuple(a, b)}
    stack_mosaics_into_3d_volume(stack_in_channel)

    min_indice = inputSlices.map{meta, _files -> meta}.min().toInteger()

    // Collect slice IDs for the pairwise transforms
    estimate_pairwise_channel = attenuation_correction.out
        .map{meta, _filename -> meta}
        .filter{v -> v as Integer != min_indice.val}
        .combine(stack_mosaics_into_3d_volume.out)
        .combine(min_indice)

    // Estimate pairwise transforms
    estimate_pairwise_transform(estimate_pairwise_channel)

    // Use transforms to align the 3D volume
    estimate_pairwise_transform.out
        .collectFile(name: 'results.csv', sort: {a, b -> a[0] <=> b[0]}, newLine: true) {["results.csv", "${it[0]},${it[1]}"]}
        .subscribe { file ->
            println "Entries are saved to file: $file"
            println "File content is: ${file.text}"
        }
}
