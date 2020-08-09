#!/usr/bin/env python3

import os
import pandas as pd

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

basepath = ""

np.random.seed(42)

numfeatlist = np.arange(5, 55, 5)

classifiernames = ["Nearest Neighbors",
                   "Linear SVC",
                   "RBF SVC",
                   "Gaussian Process",
                   "Decision Tree",
                   "Random Forest",
                   "Multilayer Perceptron",
                   "AdaBoost",
                   "Naive Bayes", 
                   "QDA", 
                   "XGBoost",
                   "Logistic Regression"
                   ]

selectornames = ["Relief", "Fisher score", "Gini index", "Chi-square", "Joint mutual information",
                 "Conditional infomax feature extraction", "Double input symmetric relevance",
                 "Mutual information maximization", "Conditional mutual information maximization",
                 "Interaction capping", "T-test", "Minimum redundancy maximum relevance",
                 "Mutual information feature selection"]
selectornames_short = ["RELF", "FSCR", "GINI", "CHSQ", "JMI", "CIFE", "DISR", "MIM", "CMIM", "ICAP", "TSCR", "MRMR",
                       "MIFS"]

# class boundary list
class_boundary_list = [304.2, 365, 425.8, 540]

# load metric files
for boundary in class_boundary_list:
    auc = pd.read_csv(os.path.join(basepath, "AUC" + str(boundary) + ".csv"), index_col="Selector")
    acc = pd.read_csv(os.path.join(basepath, "Accuracy" + str(boundary) + ".csv"), index_col="Selector")
    acc_bal = pd.read_csv(os.path.join(basepath, "Accuracy_balanced" + str(boundary) + ".csv"), index_col="Selector")
    f1 = pd.read_csv(os.path.join(basepath, "F1score" + str(boundary) + ".csv"), index_col="Selector")
    sens = pd.read_csv(os.path.join(basepath, "Sensitivity" + str(boundary) + ".csv"), index_col="Selector")
    spec = pd.read_csv(os.path.join(basepath, "Specificity" + str(boundary) + ".csv"), index_col="Selector")
    prec = pd.read_csv(os.path.join(basepath, "Precision" + str(boundary) + ".csv"), index_col="Selector")

    ax1 = sns.heatmap(auc, annot=True, linewidths=.5, cmap="magma", fmt='.2f', square=False, vmax=1, annot_kws={"size": 7})

    plt.xlabel("Machine learning model")
    plt.ylabel("Feature selector")
    plt.title("AUC - class boundary at " + str(boundary) + " days")
    plt.tight_layout()
    plt.savefig(os.path.join(basepath, "AUC" + str(boundary) + ".png"))
    plt.close()

    ax2 = sns.heatmap(acc, annot=True, linewidths=.5, cmap="magma", fmt='.2f', square=False, vmax=1, annot_kws={"size": 7})

    plt.xlabel("Machine learning model")
    plt.ylabel("Feature selector")
    plt.title("Accuracy - class boundary at " + str(boundary) + " days")
    plt.tight_layout()
    plt.savefig(os.path.join(basepath, "Accuracy" + str(boundary) + ".png"))
    plt.close()

    ax3 = sns.heatmap(acc_bal, annot=True, linewidths=.5, cmap="magma", fmt='.2f', square=False, vmax=1, annot_kws={"size": 7})

    plt.xlabel("Machine learning model")
    plt.ylabel("Feature selector")
    plt.title("Balanced accuracy - class boundary at " + str(boundary) + " days")
    plt.tight_layout()
    plt.savefig(os.path.join(basepath, "Accuracy_bal" + str(boundary) + ".png"))
    plt.close()

    ax4 = sns.heatmap(f1, annot=True, linewidths=.5, cmap="magma", fmt='.2f', square=False, vmax=1, annot_kws={"size": 7})

    plt.xlabel("Machine learning model")
    plt.ylabel("Feature selector")
    plt.title("F1 score - class boundary at " + str(boundary) + " days")
    plt.tight_layout()
    plt.savefig(os.path.join(basepath, "F1 Score" + str(boundary) + ".png"))
    plt.close()

    ax5 = sns.heatmap(sens, annot=True, linewidths=.5, cmap="magma", fmt='.2f', square=False, vmax=1, annot_kws={"size": 7})

    plt.xlabel("Machine learning model")
    plt.ylabel("Feature selector")
    plt.title("Sensitivity - class boundary at " + str(boundary) + " days")
    plt.tight_layout()
    plt.savefig(os.path.join(basepath, "Sensitivity" + str(boundary) + ".png"))
    plt.close()

    ax6 = sns.heatmap(spec, annot=True, linewidths=.5, cmap="magma", fmt='.2f', square=False, vmax=1, annot_kws={"size": 7})

    plt.xlabel("Machine learning model")
    plt.ylabel("Feature selector")
    plt.title("Specificity - class boundary at " + str(boundary) + " days")
    plt.tight_layout()
    plt.savefig(os.path.join(basepath, "Specificity" + str(boundary) + ".png"))
    plt.close()

    ax7 = sns.heatmap(prec, annot=True, linewidths=.5, cmap="magma", fmt='.2f', square=False, vmax=1, annot_kws={"size": 7})

    plt.xlabel("Machine learning model")
    plt.ylabel("Feature selector")
    plt.title("Precision - class boundary at " + str(boundary) + " days")
    plt.tight_layout()
    plt.savefig(os.path.join(basepath, "Precision" + str(boundary) + ".png"))
    plt.close()

