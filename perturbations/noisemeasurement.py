#!/usr/bin/env python3

import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import seaborn as sns

import argparse


def main(patientid: str, rootpath: str, wskullroot: str):

    rayleighcorr = 0.655

    patlist = [elem for elem in os.listdir(rootpath)] if os.path.isdir(os.path.join(rootpath, elem))
    print(patlist)
    sequences = ["T1c", "T1", "T2", "Flair"]

    noisemeasdf = pd.DataFrame(np.full([len(patlist), len(sequences)], np.NaN), columns=sequences)
    # bestmatchdf['ID'] = patlist
    noisemeasdf.index = patlist

    for pat in patlist:
        print(pat)

        # load segmentation
        seg = sitk.ReadImage(os.path.join(rootpath, pat, 'completeLabelImage.nii.gz'))

        # load registered sequences
        t1c = sitk.ReadImage(os.path.join(rootpath, pat, 'T1c-registered_reg_biascorr.nii.gz'))
        t1 = sitk.ReadImage(os.path.join(rootpath, pat, 'T1-registered_reg_biascorr.nii.gz'))
        t2 = sitk.ReadImage(os.path.join(rootpath, pat, 'T2-registered_reg_biascorr.nii.gz'))
        flair = sitk.ReadImage(os.path.join(rootpath, pat, 'Flair-registered_reg_biascorr.nii.gz'))

        # load brainmask
        bm = sitk.ReadImage(os.path.join(rootpath, pat, 'brainmask.nii.gz'))

        # load transforms
        t1ctfm = sitk.ReadTransform(os.path.join(rootpath, pat, 't1cTransform.tfm'))
        t1cinv = t1ctfm.GetInverse()

        t1tfm = sitk.ReadTransform(os.path.join(rootpath, pat, 't1Transform.tfm'))
        t1inv = t1tfm.GetInverse()

        t2tfm = sitk.ReadTransform(os.path.join(rootpath, pat, 't2Transform.tfm'))
        t2inv = t2tfm.GetInverse()

        flairtfm = sitk.ReadTransform(os.path.join(rootpath, pat, 'flairTransform.tfm'))
        flairinv = flairtfm.GetInverse()

        # load non-skullstripped images
        t1c_skull = sitk.ReadImage(os.path.join(wskullroot, pat, 'T1c.nii.gz'))
        t1_skull = sitk.ReadImage(os.path.join(wskullroot, pat, 'T1.nii.gz'))
        t2_skull = sitk.ReadImage(os.path.join(wskullroot, pat, 'T2.nii.gz'))
        flair_skull = sitk.ReadImage(os.path.join(wskullroot, pat, 'Flair.nii.gz'))


        minmaxflt = sitk.MinimumMaximumImageFilter()
        minmaxflt.Execute(t1c_skull)
        t1cmax = minmaxflt.GetMaximum()
        minmaxflt.Execute(t1_skull)
        t1max = minmaxflt.GetMaximum()
        minmaxflt.Execute(t2_skull)
        t2max = minmaxflt.GetMaximum()
        minmaxflt.Execute(flair_skull)
        flairmax = minmaxflt.GetMaximum()

        # create foreground mask, if it not already exists
        if os.path.isfile(os.path.join(wskullroot, pat, 'T1c_foregroundmask.nii.gz')) and os.path.isfile(os.path.join(wskullroot, pat, 'completelabel_t1c_skull.nii.gz')):
            t1cmask = sitk.ReadImage(os.path.join(os.path.join(wskullroot, pat, 'T1c_foregroundmask.nii.gz')))
            seg_skullspace_t1c = sitk.ReadImage(os.path.join(wskullroot, pat, 'completelabel_t1c_skull.nii.gz'))

        else:

            t1c_binthreshmask = sitk.BinaryThreshold(t1c_skull, 50, t1cmax, 1, 0)

            holefillflt = sitk.VotingBinaryHoleFillingImageFilter()
            holefillflt.SetBackgroundValue(0)
            holefillflt.SetForegroundValue(1)
            holefillflt.SetMajorityThreshold(1)
            holefillflt.SetRadius(20)
            t1cmask = holefillflt.Execute(t1c_binthreshmask)

            sitk.WriteImage(t1cmask, os.path.join(os.path.join(wskullroot, pat, 'T1c_foregroundmask.nii.gz')))

            # transform and resample label map to "skull-space" to measure noise
            resampler = sitk.ResampleImageFilter()
            resampler.SetDefaultPixelValue(0)

            # T1c segmentation - skullspace
            resampler.SetReferenceImage(t1c_skull)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            seg_skullspace_t1c = resampler.Execute(seg)
            sitk.WriteImage(seg_skullspace_t1c, os.path.join(wskullroot, pat, 'completelabel_t1c_skull.nii.gz'))


        # T1 segmenetation - skullspace
        resampler = sitk.ResampleImageFilter()
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(t1inv)
        resampler.SetReferenceImage(t1_skull)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        seg_skullspace_t1 = resampler.Execute(seg)
        t1mask = resampler.Execute(t1cmask)
        sitk.WriteImage(seg_skullspace_t1,
                        os.path.join(wskullroot, pat, 'completelabel_t1_skull.nii.gz'))
        sitk.WriteImage(t1mask,
                        os.path.join(wskullroot, pat, 'fgmask_t1_skull.nii.gz'))

        # T2 segmenetation - skullspace
        resampler.SetTransform(t2inv)
        resampler.SetReferenceImage(t2_skull)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        seg_skullspace_t2 = resampler.Execute(seg)
        t2mask = resampler.Execute(t1cmask)
        sitk.WriteImage(seg_skullspace_t2,
                        os.path.join(wskullroot, pat, 'completelabel_t2_skull.nii.gz'))
        sitk.WriteImage(t2mask,
                        os.path.join(wskullroot, pat, 'fgmask_t2_skull.nii.gz'))

        # Flair segmenetation - skullspace
        resampler.SetTransform(flairinv)
        resampler.SetReferenceImage(flair_skull)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        seg_skullspace_flair = resampler.Execute(seg)
        flairmask = resampler.Execute(t1cmask)
        sitk.WriteImage(seg_skullspace_flair,
                        os.path.join(wskullroot, pat, 'completelabel_flair_skull.nii.gz'))
        sitk.WriteImage(flairmask,
                        os.path.join(wskullroot, pat, 'fgmask_flair_skull.nii.gz'))

        # measure noise for each MR sequence separately
        stats = sitk.LabelStatisticsImageFilter()
        stats.Execute(t1c_skull, seg_skullspace_t1c)

        factor = 3.5
        t1c_wmmean = stats.GetMean(3)
        stats.Execute(t1c_skull, t1cmask)
        t1c_bgsigma = stats.GetSigma(0)
        t1c_snr = rayleighcorr * t1c_wmmean / t1c_bgsigma

        stats.Execute(t1_skull, seg_skullspace_t1)
        t1_wmmean = stats.GetMean(3)
        stats.Execute(t1_skull, t1mask)
        t1_bgsigma = stats.GetSigma(0)
        t1_snr = rayleighcorr * t1_wmmean / t1_bgsigma

        stats.Execute(t2_skull, seg_skullspace_t2)
        t2_wmmean = stats.GetMean(3)
        stats.Execute(t2_skull, t2mask)
        t2_bgsigma = stats.GetSigma(0)
        t2_snr = rayleighcorr * t2_wmmean / t2_bgsigma

        stats.Execute(flair_skull, seg_skullspace_flair)
        flair_wmmean = stats.GetMean(3)
        stats.Execute(flair_skull, flairmask)
        flair_bgsigma = stats.GetSigma(0)
        flair_snr = rayleighcorr * flair_wmmean / flair_bgsigma

        noisemeasdf.loc[pat, :] = [t1c_snr, t1_snr, t2_snr, flair_snr]
        print(noisemeasdf)


    ax = sns.boxplot(data=noisemeasdf)
    ax.set(xlabel='MR Sequence', ylabel='SNR', title='Signal-to-Noise Ratio - Robustness Subset')
    plt.show()

if __name__ == "__main__":
    """The program's entry point."""
    parser = argparse.ArgumentParser(description='Measure the noise level')

    parser.add_argument(
        '--patientid',
        type=str,
        default='GBM_01',
        help='ID of patient folder in rootpath'
    )

    parser.add_argument(
        '--rootpath',
        type=str,
        default='',
        help='Directory containt all patient subfolders with skull-stripped data'
    )

    parser.add_argument(
        '--wskullroot',
        type=str,
        default='',
        help='Directory containt all patient subfolders with original data (non-skull-stripped)'
    )

    args = parser.parse_args()

main(args.patientid, args.rootpath, args.wskullroot)
