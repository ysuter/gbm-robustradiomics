#!/usr/bin/env python3

import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import seaborn as sns

rayleighcorr = 0.655
noisemean = 0
seed = 42

patroot = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/MANAGE/data/robustness/preprocessed_segmented'
outpathvoxel = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/MANAGE/data/robustness/perturbed/voxelsize'
outpathspacing = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/MANAGE/data/robustness/perturbed/zspacing'

sequencelist = ["T1c", "T1", "T2", "Flair"]

segmentlist = ['cet', 'net', 'ed', 'ncr', 'core', 'wt', 'net_ncr', 'net_ed']

voxsizelist = np.linspace(1, 1.5, 10)
voxsizelist = np.around(voxsizelist, decimals=2)

# spacing according to the imaging recommendations (see paper for reference)
T1c_spacinglist = np.around(np.linspace(1, 1.5, 10), decimals=2)
T1_spacinglist = np.around(np.linspace(1, 1.5, 10), decimals=2)
T2_spacinglist = np.around(np.linspace(1, 4, 10), decimals=2)
Flair_spacinglist = np.around(np.linspace(1, 4, 10), decimals=2)

patlist = os.listdir(patroot)

for pat in patlist:

    patoutvoxel = os.path.join(outpathvoxel, pat)
    patoutspacing = os.path.join(outpathspacing, pat)

    if not os.path.isdir(patoutvoxel):
        os.makedirs(patoutvoxel, exist_ok=True)
    if not os.path.isdir(patoutspacing):
        os.makedirs(patoutspacing, exist_ok=True)

    # resample segmentation
    for label in segmentlist:
        seg = sitk.ReadImage(os.path.join(patroot, pat, pat + "_" + label + ".nii.gz"))
        for voxsize in voxsizelist:
            new_spacing = [voxsize, voxsize, voxsize]

            newseg = sitk.Resample(seg, seg.GetSize(),
                                             sitk.Transform(),
                                             sitk.sitkNearestNeighbor,
                                             seg.GetOrigin(),
                                             [voxsize, voxsize, voxsize],  # seg.GetSpacing(),
                                             seg.GetDirection(),
                                             0,
                                             seg.GetPixelID())

            sitk.WriteImage(newseg, os.path.join(outpathvoxel, pat, pat + "_" + label + "_isovox_" + str(voxsize) + ".nii.gz"))


    for seqidx, seq in enumerate(sequencelist):
        # load MR sequence
        img = sitk.ReadImage(os.path.join(patroot, pat, seq + "-registered_reg_biascorr.nii.gz"))

        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator = sitk.sitkBSpline
        resample.SetOutputDirection = img.GetDirection()
        resample.SetOutputOrigin = img.GetOrigin()

        orig_size = np.array(img.GetSize(), dtype=np.int)
        orig_spacing = img.GetSpacing()

        for voxsize in voxsizelist:

            new_spacing = [voxsize, voxsize, voxsize]

            newimage = sitk.Resample(img, seg.GetSize(),
                                             sitk.Transform(),
                                             sitk.sitkBSpline,
                                             seg.GetOrigin(),
                                             [voxsize, voxsize, voxsize],
                                             seg.GetDirection(),
                                             0,
                                             seg.GetPixelID())

            sitk.WriteImage(newimage, os.path.join(outpathvoxel, pat, seq + "_isovox_" + str(voxsize) + ".nii.gz"))

