#!/usr/bin/env python3

import SimpleITK as sitk
import numpy as np
import os
import pymia.evaluation.evaluator as pymia_eval
import pymia.evaluation.metric as pymia_metric
import argparse


def elasticdeform(inpimg: sitk.Image, deformation_sigma: float) -> sitk.Image:

    num_control_points= 10
    interpolator = sitk.sitkNearestNeighbor
    spatial_rank = 2
    fill_value = 0.0

    # initialize B-spline transformation
    transform_mesh_size = [num_control_points] * inpimg.GetDimension()
    bspline_transformation = sitk.BSplineTransformInitializer(inpimg, transform_mesh_size)
    params = bspline_transformation.GetParameters()
    params = np.asarray(params, dtype=np.float)
    params += np.random.randn(params.shape[0]) * deformation_sigma

    params = tuple(params)
    bspline_transformation.SetParameters(tuple(params))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(inpimg)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(fill_value)
    resampler.SetTransform(bspline_transformation)

    img_deformed = resampler.Execute(inpimg)
    img_deformed.CopyInformation(inpimg)

    return img_deformed

def main(inputdir: str, csvoutputdir: str, segname: str):

    # ! THIS PARAMETER FOR THE DEFORMATION SIGMA HAS TO BE TUNED PER LABEL TO MATCH INTERRATER-VARIABILITY ! #
    sigmaarr = np.linspace(2, 8, 31)

    subjroot = '/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/MANAGE/data/robustness/preprocessed_segmented'
    csvoutputdir = os.path.join('/media/yannick/c4a7e8d3-9ac5-463f-b6e6-92e216ae6ac0/MANAGE/data/robustness/segdeform/interrateroutput', segname)

    # make output directory if it does not already exists
    if not os.path.isdir(csvoutputdi):
        os.makedirs(csvoutputdi)

    patlist = os.listdir(subjroot)

    evaluator = pymia_eval.Evaluator(pymia_eval.ConsoleEvaluatorWriter(5))
    evaluator.add_label(1, segname)
    evaluator.add_metric(pymia_metric.DiceCoefficient())

    # for sigmaidx, sigma in enumerate(deformation_sigma):
    for sigmaval in sigmaarr:
        evaluator.add_writer(pymia_eval.CSVEvaluatorWriter(os.path.join(csvoutputdir, 'results_' + str(sigmaval) + '.csv')))
        for patidx, pat in enumerate(patlist):
            # read CET image
            img_orig = sitk.ReadImage(os.path.join(subjroot, pat, pat + '_' + segname + '.nii.gz'))
            for runidx_cet in range(0, 100):
                deformed = elasticdeform(img_orig, sigmaval)

                evaluator.evaluate(img_orig, deformed, pat + '_' + str(sigmaval) + '_' + str(runidx_cet))


if __name__ == "__main__":
    """The program's entry point."""
    parser = argparse.ArgumentParser(description='Elastic deformation for inter-rater variability simulation. Check the generated CSVs to match the desired interrater variability')

    parser.add_argument(
        '--inputdir',
        type=str,
        default='',
        help='Path to the current patients top directory.'
    )

    parser.add_argument(
        '--csvoutputdir',
        type=str,
        default='',
        help='Path where the output CSVs are saved.'
    )

    parser.add_argument(
        '--segname',
        type=str,
        default='cet',
        help='Segmentation to deform.'
    )

    args = parser.parse_args()

main(args.inputdir, args.csvoutputdir, args.segname)

