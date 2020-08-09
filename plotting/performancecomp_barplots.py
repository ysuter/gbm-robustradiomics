import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import pandas as pd
import seaborn as sns

# plt.style.use('seaborn-talk')

# plt.style.use('ggplot')
# plt.style.use('seaborn-colorblind')
sns.set_context("talk")
# load csv

offset = 0.15
annotoffset_nonrob = 0.25
annotoffset_rob = 0.16

# ### 2 classes ###
data = pd.read_csv('bestmodels_comparison.csv')

metrics_colnames = ["AUC"]

metricslist = ["AUC"]

data_curr = data.loc[:, np.append(["Prior", "Center", "Robustness", "Class Boundary"], metrics_colnames)]

my_palette = plt.cm.get_cmap("inferno", 6)

data_nonrob = data_curr.loc[(data_curr["Robustness"] == "Non-robust") & (
                    data_curr["Prior"] == "Sequence prior") & (data_curr["Class Boundary"] == 425.8), ["Center", "AUC"]]

data_rob = data_curr.loc[(data_curr["Robustness"] == "Robust") & (
                    data_curr["Prior"] == "Sequence prior") & (data_curr["Class Boundary"] == 425.8), ["Center", "AUC"]]


fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, squeeze=True, figsize=(25, 10))

markersize_circ = 6*20
markersize_star = 12*20

for metric_idx, metric in enumerate(metrics_colnames):

    # -------- non-robust, without prior ---------#
    data_single_nonrob_noprior = data_curr.loc[
        (data_curr["Center"] == "Single-center") & (data_curr["Robustness"] == "Non-robust") & (
                    data_curr["Prior"] == "None"), [metric,
                                                    "Class Boundary"]]
    data_mult_nonrob_noprior = data_curr.loc[
        (data_curr["Center"] == "Multi-center") & (data_curr["Robustness"] == "Non-robust") & (
                    data_curr["Prior"] == "None"), [metric, "Class Boundary"]]
    nonrob_noprior_delta = data_single_nonrob_noprior[metric].values - data_mult_nonrob_noprior[metric].values
    label_nonrob_noprior = [r'$\Delta$=' + "{:.2f}".format(l) for l in nonrob_noprior_delta]
    # -------- robust, without prior ---------#
    data_single_rob_noprior = data_curr.loc[
        (data_curr["Center"] == "Single-center") & (data_curr["Robustness"] == "Robust") & (
                    data_curr["Prior"] == "None"), [metric,
                                                    "Class Boundary"]]
    data_mult_rob_noprior = data_curr.loc[
        (data_curr["Center"] == "Multi-center") & (data_curr["Robustness"] == "Robust") & (
                    data_curr["Prior"] == "None"), [metric, "Class Boundary"]]
    rob_noprior_delta = data_single_rob_noprior[metric].values - data_mult_rob_noprior[metric].values
    label_rob_noprior_delta = [r'$\Delta$=' + "{:.2f}".format(l) for l in rob_noprior_delta]
    # --------non-robust, with prior ---------#
    data_single_nonrob_prior = data_curr.loc[
        (data_curr["Center"] == "Single-center") & (data_curr["Robustness"] == "Non-robust") & (
                    data_curr["Prior"] == "Sequence prior"), [metric,
                                                              "Class Boundary"]]
    data_mult_nonrob_prior = data_curr.loc[
        (data_curr["Center"] == "Multi-center") & (data_curr["Robustness"] == "Non-robust") & (
                    data_curr["Prior"] == "Sequence prior"), [metric, "Class Boundary"]]
    nonrob_prior_delta = data_single_nonrob_prior[metric].values - data_mult_nonrob_prior[metric].values
    label_nonrob_prior_delta = [r'$\Delta$=' + "{:.2f}".format(l) for l in nonrob_prior_delta]
    # --------robust, with prior ---------#
    data_single_rob_prior = data_curr.loc[
        (data_curr["Center"] == "Single-center") & (data_curr["Robustness"] == "Robust") & (
                    data_curr["Prior"] == "Sequence prior"), [metric,
                                                              "Class Boundary"]]
    data_mult_rob_prior = data_curr.loc[
        (data_curr["Center"] == "Multi-center") & (data_curr["Robustness"] == "Robust") & (
                    data_curr["Prior"] == "Sequence prior"), [metric, "Class Boundary"]]
    rob_prior_delta = data_single_rob_prior[metric].values - data_mult_rob_prior[metric].values
    label_rob_prior_delta = [r'$\Delta$=' + "{:.2f}".format(l) for l in rob_prior_delta]

    data_single_nonrob_noprior.set_index(["Class Boundary"], inplace=True)
    data_mult_nonrob_noprior.set_index(["Class Boundary"], inplace=True)
    data_single_rob_noprior.set_index(["Class Boundary"], inplace=True)
    data_mult_rob_noprior.set_index(["Class Boundary"], inplace=True)
    data_single_nonrob_prior.set_index(["Class Boundary"], inplace=True)
    data_mult_nonrob_prior.set_index(["Class Boundary"], inplace=True)
    data_single_rob_prior.set_index(["Class Boundary"], inplace=True)
    data_mult_rob_prior.set_index(["Class Boundary"], inplace=True)

    # plotting
    data_range = np.arange(1, len(data_single_nonrob_noprior.values) + 1)

    e_nonrob = ax.reshape(-1)[metric_idx].scatter([], [], s=0, label="Non-robust features, without prior")

    l_nonrob_noprior = ax.reshape(-1)[metric_idx].vlines(x=data_range - 0.5 * offset, ymin=data_mult_nonrob_noprior,
                                                         ymax=data_single_nonrob_noprior,
                                                         color=my_palette(0), alpha=0.4)
    s1_nonrob_noprior = ax.reshape(-1)[metric_idx].scatter(data_range - 0.5 * offset, data_mult_nonrob_noprior,
                                                           color=my_palette(0), alpha=0.4, marker='*', s=markersize_star,
                                                           label='Multi-center, non-robust, no prior')
    s2_nonrob_noprior = ax.reshape(-1)[metric_idx].scatter(data_range - 0.5 * offset, data_single_nonrob_noprior,
                                                           color=my_palette(0), alpha=1, marker='o', s=markersize_circ,
                                                           label='Single-center, non-robust, no prior')
    ###
    # l_nonrob_prior = ax.reshape(-1)[metric_idx].vlines(x=data_range - 0.5 * offset, ymin=data_mult_nonrob_prior,
    #                                                    ymax=data_single_nonrob_prior,
    #                                                    color=my_palette(1), alpha=0.4)
    # s1_nonrob_prior = ax.reshape(-1)[metric_idx].scatter(data_range - 0.5 * offset, data_mult_nonrob_prior,
    #                                                      color=my_palette(1), alpha=0.4, marker='*',
    #                                                      label='Multi-center, non-robust, sequence prior')
    # s2_nonrob_prior = ax.reshape(-1)[metric_idx].scatter(data_range - 0.5 * offset, data_single_nonrob_prior,
    #                                                      color=my_palette(1), alpha=1, marker='o', s=8,
    #                                                      label='Single-center, non-robust, sequence prior')
    #
    e_rob = ax.reshape(-1)[metric_idx].scatter([], [], s=0, label="Robust features, with prior")
    # l_rob_noprior = ax.reshape(-1)[metric_idx].vlines(x=data_range + 0.5 * offset, ymin=data_mult_rob_noprior,
    #                                                   ymax=data_single_rob_noprior,
    #                                                   color=my_palette(2), alpha=0.4)
    # s1_rob_noprior = ax.reshape(-1)[metric_idx].scatter(data_range + 0.5 * offset, data_mult_rob_noprior,
    #                                                     color=my_palette(2), alpha=0.4, marker='*',
    #                                                     label='Multi-center, robust, no prior')
    # s2_rob_noprior = ax.reshape(-1)[metric_idx].scatter(data_range + 0.5 * offset, data_single_rob_noprior,
    #                                                     color=my_palette(2), alpha=1, marker='o', s=8,
    #                                                     label='Single-center, robust, no prior')

    l_rob_prior = ax.reshape(-1)[metric_idx].vlines(x=data_range + 0.5 * offset, ymin=data_mult_rob_prior,
                                                    ymax=data_single_rob_prior,
                                                    color=my_palette(3), alpha=0.4)
    s1_rob_prior = ax.reshape(-1)[metric_idx].scatter(data_range + 0.5 * offset, data_mult_rob_prior,
                                                      color=my_palette(3), alpha=0.4, marker='*', s=markersize_star, label='Multi-center, robust, sequence prior')
    s2_rob_prior = ax.reshape(-1)[metric_idx].scatter(data_range + 0.5 * offset, data_single_rob_prior,
                                                      color=my_palette(3), alpha=1, marker='o', s=markersize_circ,
                                                      label='Single-center, robust, sequence prior')
    for x_arr, y_arr, annot in zip(
            data_range - annotoffset_rob*1.7, np.squeeze(data_single_nonrob_noprior.values + data_mult_nonrob_noprior.values) / 2,
            label_nonrob_noprior):
        print(x_arr)
        print(y_arr)
        print(annot)
        ax.reshape(-1)[metric_idx].annotate(xy=[x_arr, y_arr], s=annot, color=my_palette(0), rotation=90, verticalalignment='center')

    for x_arr, y_arr, annot in zip(
            data_range + annotoffset_rob*1.1, np.squeeze(data_single_rob_prior.values + data_mult_rob_prior.values) / 2,
            label_rob_prior_delta):
        print(x_arr)
        print(y_arr)
        print(annot)
        ax.reshape(-1)[metric_idx].annotate(xy=[x_arr, y_arr], s=annot, color=my_palette(3), rotation=90, verticalalignment='center')



    ax.reshape(-1)[metric_idx].set_ylim([0, 1.001])

    sns.despine(left=False, bottom=True)
    ax.reshape(-1)[metric_idx].set_xlabel("Class boundary / days")
    ax.reshape(-1)[metric_idx].set_title(metricslist[metric_idx], {'fontweight': 'bold'})
    ax.reshape(-1)[metric_idx].set_ylabel(metricslist[metric_idx])
    ax.reshape(-1)[metric_idx].set_xticks(data_range)
    ax.reshape(-1)[metric_idx].set_xticklabels(data_single_nonrob_noprior.index)

    plt.ylabel(metricslist[metric_idx])

handles, labels = ax.reshape(-1)[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='lower center', ncol=1, bbox_to_anchor=(0.525, 0.15), frameon=True, fontsize=17)  #
for t in legend.get_texts()[::3]:
    xfm = transforms.offset_copy(t.get_transform(), ax.reshape(-1)[metric_idx].figure, x=-40, units="points")
    t.set_transform(xfm)
    t._fontproperties = legend.get_texts()[-1]._fontproperties.copy()
    t.set_weight('bold')


plt.suptitle(
    "Performance of models trained on single-center data applied to multi-center data - two overall survival classes",
    y=0.99, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.93)

plt.savefig('perftest_2classes_reduced.png')
plt.savefig('perftest_2classes_reduced.svg')
plt.show()

