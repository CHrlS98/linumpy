#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Workflow Description
// Convert raw S-OCT tiles into mosaic grids and xy shifts
// Input: Directory containing raw data set tiles
// Output: Mosaic grids and xy shifts

// Parameters
params.input = ""
params.output = "output"
params.use_old_folder_structure = false // Use the old folder structure where tiles are not stored in subfolders based on their Z
params.processes = 1 // Maximum number of python processes per nextflow process
params.axial_resolution = 1.5 // Axial resolution of imaging system in microns

process create_mosaic_grid {
    input:
        tuple val(slice_id), path(tiles)
    output:
        tuple val(slice_id), path("*.ome.zarr")
    script:
    """
    linum_create_mosaic_grid_3d.py mosaic_grid_3d.ome.zarr --from_tiles_list $tiles --resolution -1 --n_processes ${params.processes} --axial_resolution ${params.axial_resolution} --n_levels 0
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
}
