#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Workflow Description
// Register 2.5d images to common space
// Input: Directory containing nifti images (RAS)
// Output: Registration results

// Parameters
params.input = "";
params.output = "results";
params.resolution = 100; // Resolution of the reconstruction in micron/pixel
params.syn_quick = false;

process mirror_image {
    input:
        tuple val(sub_id), path(image)
    output:
        tuple val(sub_id), path("${sub_id}__mirror.nii.gz")
    script:
    """
    linum_mirror_image.py $image ${sub_id}__mirror.nii.gz
    """
}

process resample {
    input:
        tuple val(sub_id), path(image)
    output:
        tuple val(sub_id), path("${sub_id}__${params.resolution}um.nii.gz")
    script:
    """
    linum_resample.py $image ${sub_id}__${params.resolution}um.nii.gz $params.resolution
    """
}

process register_to_self {
    input:
        tuple val(sub_id), path(mirror), path(original)
    output:
        tuple val(sub_id), path("${sub_id}__mirror_to_self_Warped.nii.gz")
    script:
    """
    if [ $params.syn_quick ]; then
        COMMAND=antsRegistrationSyNQuick.sh
    else
        COMMAND=antsRegistrationSyN.sh
    fi
    \$COMMAND -d 3 -t a -f $original -m $mirror -o ${sub_id}__mirror_to_self_
    """
}

process average {
    input:
    tuple val(sub_id), path(fixed), path(moving)

    output:
    tuple val(sub_id), path("${sub_id}__max.nii.gz")

    script:
    """
    linum_average_volumes.py --in_volumes $fixed $moving --out_volume "${sub_id}__max.nii.gz" --mode max
    """
}

process register_average {
    input:
        tuple val(sub_id), path(moving), path(fixed)
    output:
        tuple val(sub_id), path("${sub_id}_register_avg_Warped.nii.gz")
    script:
    """
    antsRegistration -d 3 -o [${sub_id}_register_avg_,${sub_id}_register_avg_Warped.nii.gz,${sub_id}_register_avg_invWarped.nii.gz] -m MI[${fixed},${moving},1.0,64,None,1,1] -t Rigid[0.01] -c 50x50x200x500 -s 8x4x2x1 -f 16x8x4x1 -m MI[${fixed},${moving},1.0,128,None,1,1] -t Affine[0.01] -c 50x50x200x500 -s 8x4x2x1 -f 16x8x4x1 -m MI[${fixed},${moving},1.0,256,None,1,1] -t SyN[0.000001] -c 50x50x500x1000 -s 8x4x2x1 -f 16x8x4x1 -v -w [0.01,0.99]
    """
}

process mega_mean {
    publishDir "${params.output}/${task.process}"
    input:
        path "sub_?.nii.gz"
    output:
        path "mega_mean.nii.gz"
    script:
    """
    linum_average_volumes.py --in_volumes sub_* --out_volume "mega_mean.nii.gz" --mode mean
    """
}

workflow {
    input = Channel.fromPath("$params.input/**/*.nii.gz").map{it -> [it.parent.name, it]}

    channel_resampled = resample(input)

    channel_mirrors = mirror_image(channel_resampled)
    pairs_for_register = channel_mirrors.join(channel_resampled)

    channel_mirrors_registered = register_to_self(pairs_for_register)
    pairs_for_average = channel_mirrors_registered.join(channel_resampled)

    average(pairs_for_average)

    average_for_template = average.out
        .combine(average.out.first().flatMap{ it -> it[1]})
        .filter{it -> it[1] != it[2]}

    register_average(average_for_template)

    megamax_input = average.out.first()
    .flatMap{it -> it[1]}
    .concat(register_average.out.flatMap{ it -> it[1] })
    .collect()

    mega_mean(megamax_input)
}

