#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Workflow Description
// Register 2.5d images to common space
// Input: Directory containing nifti images (RAS)
// Output: Registration results

// Parameters
params.input = "";
params.slicing_direction = "1.0 1.0 0.0";
params.slicing_destination_axis = "x+";
params.mid_slice_index_moving = -1;
params.output = "results";
params.resolution = 50; // Resolution of the reconstruction in micron/pixel


process create_reference_image {
    input:
        tuple val(meta), path(fa), path(fodf)
    output:
        tuple val(meta), path("simulated_oct.nii.gz")
    script:
    """
    linum_simulate_oct_from_dmri.py $fodf $fa simulated_oct.nii.gz --slicing_direction $params.slicing_direction --dest_axis $params.slicing_destination_axis --bg_threshold 0.0 --sharpening 2.0
    """
}

process split_nifti {
    input:
        tuple val(meta), path(image)
    output:
        tuple val(meta), path("${image.getSimpleName()}__split/${meta}_slice_*.nii.gz")
    script:
    """
    linum_split_nifti.py $image ${image.getSimpleName()}__split --output_prefix ${meta}_slice_
    """
}

process register_one_to_many {
    input:
        tuple val(meta), path(image), path("ref_slice_*.nii.gz")
    output:
        tuple val(meta), path("${meta}__to_*_Warped.nii.gz")
    script:
    """
    for slice in ref_slice_*.nii.gz; do
        slice_name=\$(basename -s .nii.gz \$slice)
        antsRegistrationSyN.sh -d 2 -t a -m $image -f \${slice} -o ${meta}__to_\${slice_name}_
    done
    """
}

process register_pairwise {
    publishDir "$params.output/$task.process"
    input:
        tuple path(moving), path(fixed)
    output:
        tuple path(fixed), path("${moving.getSimpleName()}__to_${fixed.getSimpleName()}_Warped.nii.gz"), env("corrcoef")
    script:
    """
    antsRegistrationSyN.sh -d 2 -t a -m $moving -f $fixed -o ${moving.getSimpleName()}__to_${fixed.getSimpleName()}_
    corrcoef=`linum_correlate_2d_slices.py ${moving.getSimpleName()}__to_${fixed.getSimpleName()}_Warped.nii.gz $fixed`
    """
}

process resample_reference {
    input:
        tuple val(meta), path(image)
    output:
        tuple val(meta), path("${image.getSimpleName()}__200x10x10um.nii.gz")
    script:
    """
    linum_resample.py $image ${image.getSimpleName()}__200x10x10um.nii.gz 200 10 10 --interpolation nearest
    """
}

process make_three_dimensional {
    publishDir "$params.output/$task.process"
    input:
        tuple path(fixed), path(moving)
    output:
        path("${moving.getSimpleName()}__expanded.nii.gz")
    script:
    """
    linum_nifti_2d_to_reference.py $moving $fixed ${moving.getSimpleName()}__expanded.nii.gz
    """
}

workflow {
    if (params.mid_slice_index_moving == -1) {
        error "Please provide a mid slice index for the moving image with --mid_slice_index_moving."
    }

    in_oct = Channel.fromPath("$params.input/*oct.nii.gz").map{it -> ["oct", it]}
    in_fa = Channel.fromPath("$params.input/*fa.nii.gz")
    in_fodf = Channel.fromPath("$params.input/*fodf.nii.gz")

    channel_for_ref = in_fa.concat(in_fodf).collect().map{it -> ["ref", it[0], it[1]]}

    channel_ref = create_reference_image(channel_for_ref)
    channel_resampled = resample_reference(channel_ref)

    channel_for_split = in_oct.concat(channel_resampled)
    channel_split = split_nifti(channel_for_split)

    branch = channel_split.branch{it ->
        ref: it[0] == "ref"
        oct: it[0] == "oct"
    }
    all_oct_slices = branch.oct.map{it -> it[1]}.flatten()

    // Emit the oct slice corresponding to the mid slice index
    mid_slice = all_oct_slices.filter{it -> it.getSimpleName().contains("slice_${params.mid_slice_index_moving}")}
    // mid_slice.view()

    // Order is (moving, fixed)
    pairs = mid_slice.combine(branch.ref.map{it -> it[1]}.flatten())
    // pairs.view()

    // Register the mid slice to all ref slices
    // register_pairwise(pairs)

    // best_fit = register_pairwise.output
    //     .toSortedList{a, b -> b[2] <=> a[2]}
    //     .flatten()
    //     .collate(3)
    //     .take(3)
    //     .map{it -> [it[0], it[1]]}
    // make_three_dimensional(best_fit)
}

