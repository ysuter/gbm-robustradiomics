#!/usr/bin/env python3

import argparse
import SimpleITK as sitk
import os
import glob


def main(patdir: str):

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    # loop over all subdirectories / timepoints in current patient folder
    with os.scandir(patdir) as it:
        for entry in it:
            if entry.is_dir() and not entry.name.startswith('.'):
                print(entry.name)

                # loop over all nifti files and do a bias field correction
                niftifiles = glob.glob(os.path.join(patdir, entry.name, '*.nii.gz'))
                for currnifti in niftifiles:
                    if "Label" not in currnifti and "mask" not in currnifti and "CET" not in currnifti:
                        print(currnifti)
                        imgpath, _ = currnifti.split('.nii.gz')
                        imgname = imgpath + '-biascorr.nii.gz'
                        print(imgname)
                        img = sitk.Cast(sitk.ReadImage(currnifti), sitk.sitkFloat32)
                        img_pixeltype = img.GetPixelIDValue()
                        img_corr = corrector.Execute(img)
                        img_corr = sitk.Cast(img_corr, img_pixeltype)

                        sitk.WriteImage(img_corr, imgname)
    it.close()


if __name__ == "__main__":
    """The program's entry point."""
    parser = argparse.ArgumentParser(description='Run N4 bias field correction')

    parser.add_argument(
        '--patdir',
        type=str,
        help='Path to the current patients top directory.'
    )

    args = parser.parse_args()

main(args.patdir)
