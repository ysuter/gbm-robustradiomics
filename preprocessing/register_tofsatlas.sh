#!/bin/bash

export FREESURFER_HOME=/usr/local/freesurfer
source /usr/local/freesurfer/SetUpFreeSurfer.sh

cd [PATIENT DIR]
basepath=[PATIENT DIR]

for d in */; do
    # check if current subject was already processed
    if [ ! -d ../Reglabels/${d}/ ]; then
		mkdir -p ../Reglabels/${d}
	fi

    mri_robust_register --mov ${d}/T1c-registered_reg_biascorr.nii.gz --dst /usr/local/freesurfer/subjects/fsaverage/mri/brain.mgz --lta ../Reglabels/${d}_affine.lta --iscale --initorient --affine --satit --maxit 200
    mri_convert --resample_type nearest --apply_transform ../Reglabels/${d}_affine.lta ${d}/tumorLabelImage.nii.gz ../Reglabels/${d}_segreg.nii.gz
    mri_convert --resample_type interpolate --apply_transform ../Reglabels/${d}_affine.lta ${d}/T1c-registered_reg_biascorr.nii.gz ../Reglabels/${d}_T1c-atlasreg.nii.gz

    echo "processed $d"

done
