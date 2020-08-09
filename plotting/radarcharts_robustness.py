#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np
from numpy import nan
from itertools import compress

df = pd.read_csv('', quotechar="'")
print(df)
# df.drop(columns=["ICC", "ICCupper"], inplace=True)

test = df.pivot(index='Type', columns='Perturbation', values='ICClower')

df = test

# reorder index sequence
df = df.loc[['firstorder', 'glcm', 'gldm', 'glrlm', 'glszm', 'ngtdm', 'shape', 'beteta', 'centroid', 'l6', 'l7'], :]


def make_spider(row, title, color):

    values = df.iloc[row].values.flatten().tolist()

    categories = np.unique(df.iloc[row].index.values)

    categories = categories[~np.isnan(values)]
    values = list(compress(values, ~np.isnan(values)))

    N = len(categories)

    # What will be the angle of each axis in the plot?
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(4, 3, row + 1, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axis per variable + add labels labels yet
    xlab = categories
    xlab = np.where(xlab == "noise", "Noise", xlab)
    xlab = np.where(xlab == "voxelsize", "Voxel size", xlab)
    xlab = np.where(xlab == "samplingrate", "k-space subsampling", xlab)
    xlab = np.where(xlab == "bindwidth", "Bin width", xlab)
    xlab = np.where(xlab == "deform_sigma", "Inter-rater", xlab)
    xlab = np.where(xlab == "zspacing", "Slice spacing", xlab)

    plt.xticks(angles[:-1], xlab, color='grey', size=38)
    ax.tick_params(axis='x', which='major', pad=40)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.5, 0.75, 1], ["0.5", "0.75", "1"], color="grey", size=35)
    plt.ylim(0.2, 1)
    plt.ylabel("ICC(2,1)", size=43, color="grey", labelpad=20)

    # Ind1
    print('----')
    print(categories)
    print(values)
    values += values[:1]
    # print(values)
    print('----')
    print(df.index[row])

    ax.plot(angles, values, color=color, linewidth=6, linestyle='solid')
    ax.fill(angles, values, color='green', alpha=0.3)

    # Add a title
    plt.title(title, pad=-30, size=43, color=color, y=1.2)

# ------- PART 2: Apply to all individuals
# initialize the figure
my_dpi = 96
plt.figure(figsize=(3900 / my_dpi, 4000 / my_dpi), dpi=my_dpi)
plt.subplots_adjust(hspace=0.5, wspace=0.2)

titles = df.index
titles = np.where(titles == "firstorder", "First order", titles)
titles = np.where(titles == "glcm", "GLCM", titles)
titles = np.where(titles == "glszm", "GLSZM", titles)
titles = np.where(titles == "gldm", "GLRLM", titles)
titles = np.where(titles == "glrlm", "GLDM", titles)
titles = np.where(titles == "ngtdm", "NGTDM", titles)
titles = np.where(titles == "shape", "Shape", titles)
titles = np.where(titles == "beteta", "Enhancement geometry", titles)
titles = np.where(titles == "l6", "Deep - layer 6", titles)
titles = np.where(titles == "l7", "Deep - layer 7", titles)
titles = np.where(titles == "centroid", "Centroid", titles)

print("len df index")
print(len(df.index))
# Loop to plot
for row in range(0, len(df.index)):
    # make_spider(row=row, title=titles[row], color=my_palette(9))
    make_spider(row=row, title=titles[row], color='black')

plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=7)
plt.savefig('./radarplot.png')
plt.savefig('./radarplot.svg')
plt.show()


