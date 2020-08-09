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

patroot = ''
outpath = ''

sequencelist = ["T1c", "T1", "T2", "Flair"]

minsnr = [25.5116552159421, 4.96434080433457, 5.51851313687576, 2.4366468819658]


patlist = os.listdir(patroot)

# create noise images
sigmamax = 130
sigmalist = np.linspace(1,sigmamax, 100)
sigmalist = np.around(sigmalist, decimals=2)

for pat in patlist:

    seg = sitk.ReadImage(os.path.join(patroot, pat, "completeLabelImage.nii.gz"))

    patout = os.path.join(outpath, pat)
    if not os.path.isdir(patout):
        os.makedirs(patout, exist_ok=True)

    for seqidx, seq in enumerate(sequencelist):
        # load MR sequence
        img = sitk.ReadImage(os.path.join(patroot, pat, seq + "-registered_reg_biascorr.nii.gz"))

        snrtable = np.empty([len(sigmalist), 3])
        for noiseidx, noiselevel in enumerate(sigmalist):
            noiseimg = sitk.AdditiveGaussianNoise(img, noiselevel, noisemean, seed)

            # measure noise
            stats = sitk.LabelStatisticsImageFilter()
            stats.Execute(noiseimg, seg)
            wmmean = stats.GetMean(3)
            bgsigma = stats.GetSigma(0)
            snr = rayleighcorr * wmmean / bgsigma

            currline = np.append(np.append(noiselevel, snr), abs(snr-minsnr[seqidx]))

            if noiseidx == 0:
                snrtable = currline
            else:
                snrtable = np.vstack((snrtable, currline))

        # get minimum score per row
        minrowidx = np.where(snrtable[:, -1] == np.amin(snrtable[:, -1]))

        print(minrowidx[0])
        maxsigma = snrtable[minrowidx, 0][0][0]
        print("Best match for " + sequencelist[seqidx])
        print(maxsigma)
        print(snrtable[minrowidx, :])

        defsigmalist = np.around(defsigmalist, decimals=2)

        for sigmaidx, sigmaitem in enumerate(defsigmalist):

            noiseimgpert = sitk.AdditiveGaussianNoise(img, sigmaitem, noisemean, seed)

            # measure noise
            stats = sitk.LabelStatisticsImageFilter()
            stats.Execute(noiseimgpert, seg)
            wmmean = stats.GetMean(3)
            bgsigma = stats.GetSigma(0)
            snr = rayleighcorr * wmmean / bgsigma
            sitk.WriteImage(noiseimgpert, os.path.join(outpath, pat, seq + "_sigma_" + str(sigmaitem) + "_snr_" + str(round(snr, 2)) + ".nii.gz"))
