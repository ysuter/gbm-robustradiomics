import SimpleITK as sitk
import numpy as np
import argparse
import os
import pymia.filtering.misc as flt

def round_to_mult235(inparr: np.array):

    outarr = np.empty_like(inparr)

    for idx, elem in enumerate(inparr):

        if elem % 2 == 0 or elem % 3 == 0 or elem % 5 == 0:
            outarr[idx] = elem

        else:
            mul2 = int(elem / 2) * 2 + 2
            mul3 = int(elem / 3) * 3 + 3
            mul5 = int(elem / 5) * 5 + 5

            # output smallest to avoid useless padding
            outarr[idx] = np.min([mul2, mul3, mul5])

    return outarr

def main(imginppath: str, samplerate: float, outputsubdir: str):

    samplerate_str = str("{:.2f}".format(samplerate))

    print(samplerate_str)
    print('------')

    outfilename = os.path.split(imginppath)[1].split('.nii.gz')[0] + '-subsampled_' + samplerate_str + '.nii.gz'
    outfilename = os.path.join(os.path.split(imginppath)[0], outputsubdir, outfilename)

    # load image
    img = sitk.ReadImage(imginppath)
    print(img.GetSize())

    # pad image
    padfilter = flt.SizeCorrectionFilter(two_sided=False, pad_constant=0.0)
    padparams = flt.SizeCorrectionParams(img.GetSize())

    sitkpadflt = sitk.FFTPadImageFilter()
    sitkpadflt.SetBoundaryCondition(1)

    img_padded = sitkpadflt.Execute(img)


    brainmask = sitk.ReadImage(os.path.join(os.path.split(imginppath)[0],'brainmask.nii.gz'))
    # transform to k-space
    print('Unpadded: ' + str(img.GetSize()))
    print('Padded: ' + str(img_padded.GetSize()))
    img_k = sitk.ForwardFFT(img_padded)

    # get size of transformed image
    img_k_size = img_k.GetSize()

    # random sample
    numel = img_k_size[0] * img_k_size[1] * img_k_size[2]
    samplearr = np.zeros(numel, dtype=int)
    samplearr[:int(numel*samplerate)] = 1
    np.random.shuffle(samplearr)

    maskarr = np.reshape(samplearr, img_k_size)

    maskimg = sitk.GetImageFromArray(np.swapaxes(maskarr, 0, 2))
    maskimg.CopyInformation(img_k)

    masked_kspace = sitk.Mask(img_k, maskimg)

    img_subsampled = sitk.InverseFFT(masked_kspace)

    # unpad
    unpadparams = flt.SizeCorrectionParams(img.GetSize())
    img_subsampled = padfilter.execute(img_subsampled, unpadparams)
    img_subsampled.CopyInformation(img)

    # mask with brainmask
    img_subsampled = sitk.Mask(sitk.Cast(img_subsampled, sitk.sitkInt64), sitk.Cast(brainmask, sitk.sitkInt64))

    img_subsampled = sitk.Cast(img_subsampled, img.GetPixelID())

    sitk.WriteImage(img_subsampled, outfilename)


if __name__ == "__main__":
    """The program's entry point."""
    parser = argparse.ArgumentParser(description='K-space subsampling')

    parser.add_argument(
        '--imgpath',
        type=str,
        default='',
        help='Path to the image to be subsampled.'
    )

    parser.add_argument(
        '--samplerate',
        type=float,
        default=0.90,
        help='Subsampling rate in frequency / k-space'
    )
    parser.add_argument(
        '--outputsubdir',
        type=str,
        default='subsampled',
        help='Subfolder where the subsampled images will be saved.'
    )

    args = parser.parse_args()

main(args.imgpath, args.samplerate, args.outputsubdir)





