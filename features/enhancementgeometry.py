#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
import argparse


def cet_shpericalrimwidth(pyradfeat: pd.DataFrame) -> pd.Series:
    cetfilt = pyradfeat.loc[pyradfeat['Seg'] == 'cet']
    ncrfilt = pyradfeat.loc[pyradfeat['Seg'] == 'ncr']

    cetvols = cetfilt.loc[:, 'original_shape_VoxelVolume']
    ncrvols = ncrfilt.loc[:, 'original_shape_VoxelVolume']

    ncr_zerofill = ncrvols.fillna(0)
    cet_zerofill = cetvols.fillna(0)


    outvals = 0.62*(np.subtract(np.cbrt(np.add(cet_zerofill.values, ncr_zerofill.values)), np.cbrt(ncrvols.fillna(0).values)))

    return pd.Series(outvals)

def surface(pyradfeat: pd.DataFrame) -> pd.Series:
    cetfilt = pyradfeat.loc[pyradfeat['Seg'] == 'cet']
    ncrfilt = pyradfeat.loc[pyradfeat['Seg'] == 'ncr']

    cetmeshvols = cetfilt.loc[:, 'original_shape_SurfaceArea']
    ncrmeshvols = ncrfilt.loc[:, 'original_shape_SurfaceArea']

    cetmeshvols_zerofill = cetmeshvols.fillna(0).values
    ncrmeshvols_zerofill = ncrmeshvols.fillna(0).values

    return pd.Series(np.add(cetmeshvols_zerofill, ncrmeshvols_zerofill))


def cet_surface(pyradfeat: pd.DataFrame) -> pd.Series:
    cetfilt = pyradfeat.loc[pyradfeat['Seg'] == 'cet']

    cetmeshsurf = cetfilt.loc[:, 'original_shape_SurfaceArea']

    cetmeshsurf_zerofill = cetmeshsurf.fillna(0)

    return pd.Series(cetmeshsurf_zerofill.values)


def cet_volume(pyradfeat: pd.DataFrame) -> pd.Series:
    cetfilt = pyradfeat.loc[pyradfeat['Seg'] == 'cet']

    cetvols = cetfilt.loc[:, 'original_shape_VoxelVolume']

    cetvols_zerofill = cetvols.fillna(0)

    return pd.Series(cetvols_zerofill.values)


def ncr_volume(pyradfeat: pd.DataFrame) -> pd.Series:
    ncrfilt = pyradfeat.loc[pyradfeat['Seg'] == 'ncr']

    ncrvols = ncrfilt.loc[:, 'original_shape_VoxelVolume']

    ncrvols_zerofill = ncrvols.fillna(0)

    return pd.Series(ncrvols.values)


def cet_max3ddiam(pyradfeat: pd.DataFrame) -> pd.Series:
    cetfilt = pyradfeat.loc[pyradfeat['Seg'] == 'cet']

    max3ddiam = cetfilt.loc[:, 'original_shape_Maximum3DDiameter']

    cetfilt_zerofill = max3ddiam.fillna(0)

    return pd.Series(cetfilt_zerofill.values)


def cet_surfregularity(pyradfeat: pd.DataFrame) -> pd.Series:
    cetvol = cet_volume(pyradfeat).values
    cetsurf = cet_surface(pyradfeat).values

    # print(cetvol)
    # print('----------------------------')
    # print(cetsurf)
    #
    # for idx, s in enumerate(cetsurf):
    #     print('cetvol: ' + str(cetvol[idx]))
    #     print('cetsurf: ' + str(s))
    #     print('cetsurf^3: ' + str(np.power(s, 3)))
    #     print('sqrt_cetsurf^3: ' + str(np.sqrt(np.power(s, 3))))

    return pd.Series(6*np.sqrt(np.pi) * np.divide(cetvol, np.sqrt(np.power(cetsurf, 3))))


def total_volume(pyradfeat: pd.DataFrame) -> pd.Series:
    cetvol = cet_volume(pyradfeat).values
    ncrvol = ncr_volume(pyradfeat).values
    ncrvol = np.nan_to_num(ncrvol)

    return pd.Series(np.add(cetvol, ncrvol))


def main(inpcsvpath: str, keepcols: list):
    pyradinp = pd.read_csv(inpcsvpath)

    outdf = pyradinp.loc[:, keepcols]
    outdf.drop_duplicates(['ID', 'voxelsize'],  keep='first', inplace=True)
    outdf.reset_index(drop=False, inplace=True)

    outdf = outdf.assign(NCRvolume=ncr_volume(pyradinp))
    outdf = outdf.assign(CETvolume=cet_volume(pyradinp))
    outdf = outdf.assign(TOTALvolume=total_volume(pyradinp))
    outdf = outdf.assign(CETrimwidth=cet_shpericalrimwidth(pyradinp))
    outdf = outdf.assign(CETsurface=cet_surface(pyradinp))
    outdf = outdf.assign(CETsurfacereg=cet_surfregularity(pyradinp))
    outdf = outdf.assign(CETmax3Ddiam=cet_max3ddiam(pyradinp))

    outdf = outdf.fillna(0)

    outdf.to_csv('/home/yannick/Desktop/test/molina_voxelsize_t1c.csv', index=False)


if __name__ == "__main__":
    """The program's entry point."""
    parser = argparse.ArgumentParser(description='Enhancement geometry features.')

    parser.add_argument(
        '--pyradcsv',
        type=str,
        # default='cet',
        help='Input csv with pyradimoics features, original feature names as column headers'
    )
    parser.add_argument(
        '--keepcols',
        type=list,
        default=['ID', 'Image', 'Mask', 'Label', 'Sequence'],
        help='List columns which should be kept from the inpus csv'
    )

    args = parser.parse_args()

main(args.pyradcsv, args.keepcols)

