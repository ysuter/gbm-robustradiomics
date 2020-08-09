#!/usr/bin/env python3

import os
import pandas as pd
import argparse
import numpy as np
from shutil import copyfile
from sklearn.model_selection import StratifiedKFold
# from skfeature.function import similarity_based, information_theoretical_based, statistical_based

from skfeature.function.similarity_based import reliefF, fisher_score
from skfeature.function.statistical_based import chi_square, t_score, gini_index
from skfeature.function.information_theoretical_based import CIFE, JMI, DISR, MIM, CMIM, ICAP, MRMR, MIFS

from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

twoclass_basepath = ""
twoclass_selected = ""
aucsavepath = ""

np.random.seed(42)

numfeatlist = np.arange(5, 55, 5)

classifiernames = ["Nearest Neighbors",  #
                   "Linear SVC",  #
                   "RBF SVC",
                   "Gaussian Process",  #
                   "Decision Tree",  #
                   "Random Forest",  #
                   "Multilayer Perceptron",  #
                   "AdaBoost",  #
                   "Naive Bayes",  #
                   "QDA",  #
                   "XGBoost",  #
                   "Logistic Regression"  #
                   ]

selectornames = ["Relief", "Fisher score", "Gini index", "Chi-square", "Joint mutual information",
                 "Conditional infomax feature extraction", "Double input symmetric relevance",
                 "Mutual information maximization", "Conditional mutual information maximization",
                 "Interaction capping", "T-test", "Minimum redundancy maximum relevance",
                 "Mutual information feature selection"]
selectornames_short = ["RELF", "FSCR", "GINI", "CHSQ", "JMI", "CIFE", "DISR", "MIM", "CMIM", "ICAP", "TSCR", "MRMR",
                       "MIFS"]

# class boundary list
class_boundary_list = [240, 304.2, 365, 425.8, 540, 730]
numsplits = 10
for class_boundary in class_boundary_list:

    # Dataframe with best AUC values for combinatios of selectors and models
    bestacc = pd.DataFrame(np.ones([len(selectornames_short), len(classifiernames)+1])*np.NaN, columns=["Selector"] + classifiernames)
    bestacc["Selector"] = selectornames_short
    bestacc.index = bestacc.Selector
    bestacc.drop(columns="Selector", inplace=True)

    accuracy_overall = bestacc.copy(deep=True)
    auc_overall = bestacc.copy(deep=True)
    auc_overall_NAN = bestacc.copy(deep=True)
    accuracy_overall_balanced = bestacc.copy(deep=True)
    precision_overall = bestacc.copy(deep=True)
    sensitivity_overall = bestacc.copy(deep=True)
    specificity_overall = bestacc.copy(deep=True)
    f1score_overall = bestacc.copy(deep=True)

    y_gt = []
    for currsplit in range(0, numsplits):
        # print(currsplit)
        y_gt_currsplit_path = os.path.join(twoclass_basepath, "ytest_2class_split_boundary" + str(class_boundary)
                                           + "_split" + str(currsplit)) + '.csv'
        y_gt_currsplit = pd.read_csv(y_gt_currsplit_path)
        y_gt.append(y_gt_currsplit["Survival"].values)

    for sel_idx, selector in enumerate(selectornames_short):
        for clf in classifiernames:
            if clf is "Nearest Neighbors":
                numneighbors = np.arange(3, 22, 3)
                acclist = np.zeros((len(numneighbors), len(numfeatlist)))

                # loop across number of neighbors
                for neighidx, num_n in enumerate(numneighbors):
                    # loop over number of features
                    for nfeatidx, nfeat in enumerate(numfeatlist):
                        acc_split = []
                        for currsplit in range(0, numsplits):
                            y_test_curr_path = os.path.join(twoclass_basepath, str(class_boundary), "ypred_" + selector
                                                            + "_" + clf + '_numN' + str(num_n) + "_split" + str(currsplit)
                                                            + "_numfeat" + str(nfeat) + '.csv')

                            y_testpred = pd.read_csv(y_test_curr_path)["Survival"].values

                            acc_split.append(balanced_accuracy_score(y_gt[currsplit], y_testpred))

                        acclist[neighidx, nfeatidx] = np.mean(acc_split)

                i, j = np.unravel_index(acclist.argmax(), acclist.shape)
                # print(auclist[i, j])
                # print(str(i) + ', ' + str(j))
                # print(auclist)

                best_numN = numneighbors[i]
                best_numFeat = numfeatlist[j]

                # copy best combination to new folder and calculate metrics
                acc_best_split = []
                auc_best_split = []
                acc_balanced_best_split = []
                sens_best_split = []
                spec_best_split = []
                prec_best_split = []
                f1_best_split = []

                for copysplit in range(0, numsplits):
                    source_best_y = os.path.join(twoclass_basepath, str(class_boundary), "ypred_" + selector
                                                 + "_" + clf + '_numN' + str(best_numN) + "_split"
                                                 + str(copysplit) + "_numfeat" + str(best_numFeat) + '.csv')
                    dest_best_y = os.path.join(twoclass_selected, str(class_boundary), "ypred_" + selector
                                                 + "_" + clf + '_numN' + str(best_numN) + "_split"
                                                 + str(copysplit) + "_numfeat" + str(best_numFeat) + '.csv')
                    copyfile(source_best_y, dest_best_y)

                    y_pred_curr = pd.read_csv(dest_best_y)["Survival"].values
                    y_score_curr = pd.read_csv(dest_best_y)["Score"].values

                    y_gt_currsplit = y_gt[copysplit]

                    acc_best_split.append(accuracy_score(y_gt_currsplit, y_pred_curr))
                    acc_balanced_best_split.append(balanced_accuracy_score(y_gt_currsplit, y_pred_curr))
                    f1_best_split.append(f1_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))

                    try:
                        auc_best_split.append(roc_auc_score(y_gt_currsplit, y_score_curr))
                    except:
                        auc_best_split.append(np.NaN)

                    tn, fp, fn, tp = confusion_matrix(y_gt_currsplit, y_pred_curr, labels=[0, 1]).ravel()
                    # sens.append(tp / (tp + fn))
                    sens_best_split.append(recall_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    try:
                        spec_best_split.append(tn / (tn + fp))
                    except:
                        spec_best_split.append(np.NaN)
                    prec_best_split.append(precision_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    # prec.append(tp / (tp + fp))

                auc_overall.loc[selector, clf] = np.nanmean(auc_best_split)
                auc_overall_NAN.loc[selector, clf] = np.isnan(auc_best_split).sum()
                accuracy_overall.loc[selector, clf] = np.nanmean(acc_best_split)
                accuracy_overall_balanced.loc[selector, clf] = np.nanmean(acc_balanced_best_split)
                sensitivity_overall.loc[selector, clf] = np.nanmean(sens_best_split)
                specificity_overall.loc[selector, clf] = np.nanmean(spec_best_split)
                precision_overall.loc[selector, clf] = np.nanmean(prec_best_split)
                f1score_overall.loc[selector, clf] = np.nanmean(f1_best_split)

            elif clf is "Linear SVC":
                paramspace = [0.25, 0.5, 1, 2, 4]
                acclist = np.zeros((len(paramspace), len(numfeatlist)))
                for paramidx, param in enumerate(paramspace):
                    # loop over number of features
                    for nfeatidx, nfeat in enumerate(numfeatlist):
                        acc_split = []
                        for currsplit in range(0, numsplits):
                            y_test_curr_path = os.path.join(twoclass_basepath, str(class_boundary), "ypred_" + selector
                                                            + "_" + clf + '_C' + str(param) + "_split"
                                                            + str(currsplit) + "_numfeat" + str(nfeat) + '.csv')

                            y_testpred = pd.read_csv(y_test_curr_path)["Survival"].values

                            acc_split.append(balanced_accuracy_score(y_gt[currsplit], y_testpred))

                        acclist[paramidx, nfeatidx] = np.mean(acc_split)

                i, j = np.unravel_index(acclist.argmax(), acclist.shape)

                best_param = paramspace[i]
                best_numFeat = numfeatlist[j]

                # copy best combination to new folder and calculate metrics
                acc_best_split = []
                auc_best_split = []
                acc_balanced_best_split = []
                sens_best_split = []
                spec_best_split = []
                prec_best_split = []
                f1_best_split = []

                # copy best combination to new folder
                for copysplit in range(0, numsplits):
                    source_best_y = os.path.join(twoclass_basepath, str(class_boundary), "ypred_" + selector
                                                 + "_" + clf + '_C' + str(best_param) + "_split"
                                                 + str(copysplit) + "_numfeat" + str(best_numFeat) + '.csv')
                    dest_best_y = os.path.join(twoclass_selected, str(class_boundary), "ypred_" + selector
                                               + "_" + clf + '_C' + str(best_param) + "_split"
                                               + str(copysplit) + "_numfeat" + str(best_numFeat) + '.csv')

                    copyfile(source_best_y, dest_best_y)
                    y_pred_curr = pd.read_csv(dest_best_y)["Survival"].values
                    y_score_curr = pd.read_csv(dest_best_y)["Score"].values

                    y_gt_currsplit = y_gt[copysplit]

                    acc_best_split.append(accuracy_score(y_gt_currsplit, y_pred_curr))
                    acc_balanced_best_split.append(balanced_accuracy_score(y_gt_currsplit, y_pred_curr))
                    f1_best_split.append(f1_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    try:
                        auc_best_split.append(roc_auc_score(y_gt_currsplit, y_score_curr))
                    except:
                        auc_best_split.append(np.NaN)
                    tn, fp, fn, tp = confusion_matrix(y_gt_currsplit, y_pred_curr, labels=[0, 1]).ravel()
                    # sens.append(tp / (tp + fn))
                    sens_best_split.append(recall_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    try:
                        spec_best_split.append(tn / (tn + fp))
                    except:
                        spec_best_split.append(np.NaN)
                    prec_best_split.append(precision_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    # prec.append(tp / (tp + fp))

                auc_overall.loc[selector, clf] = np.nanmean(auc_best_split)
                auc_overall_NAN.loc[selector, clf] = np.isnan(auc_best_split).sum()
                accuracy_overall.loc[selector, clf] = np.nanmean(acc_best_split)
                accuracy_overall_balanced.loc[selector, clf] = np.nanmean(acc_balanced_best_split)
                sensitivity_overall.loc[selector, clf] = np.nanmean(sens_best_split)
                specificity_overall.loc[selector, clf] = np.nanmean(spec_best_split)
                precision_overall.loc[selector, clf] = np.nanmean(prec_best_split)
                f1score_overall.loc[selector, clf] = np.nanmean(f1_best_split)

            elif clf is "Decision Tree":
                paramspace = [5, 10, 15, 20]
                acclist = np.zeros((len(paramspace), len(numfeatlist)))
                for paramidx, param in enumerate(paramspace):
                    # loop over number of features
                    for nfeatidx, nfeat in enumerate(numfeatlist):
                        acc_split = []
                        for currsplit in range(0, numsplits):
                            y_test_curr_path = os.path.join(twoclass_basepath, str(class_boundary), "ypred_" + selector
                                                            + "_" + clf + '_maxd' + str(param) + "_split"
                                                            + str(currsplit) + "_numfeat" + str(nfeat) + '.csv')

                            y_test = pd.read_csv(y_test_curr_path)["Score"].values
                            y_testpred = pd.read_csv(y_test_curr_path)["Survival"].values
                            acc_split.append(balanced_accuracy_score(y_gt[currsplit], y_testpred))

                        acclist[paramidx, nfeatidx] = np.mean(acc_split)

                i, j = np.unravel_index(acclist.argmax(), acclist.shape)

                best_param = paramspace[i]
                best_numFeat = numfeatlist[j]

                # copy best combination to new folder and calculate metrics
                acc_best_split = []
                auc_best_split = []
                acc_balanced_best_split = []
                sens_best_split = []
                spec_best_split = []
                prec_best_split = []
                f1_best_split = []

                # copy best combination to new folder
                for copysplit in range(0, numsplits):
                    source_best_y = os.path.join(twoclass_basepath, str(class_boundary), "ypred_" + selector
                                                 + "_" + clf + '_maxd' + str(best_param) + "_split"
                                                 + str(copysplit) + "_numfeat" + str(best_numFeat) + '.csv')
                    dest_best_y = os.path.join(twoclass_selected, str(class_boundary), "ypred_" + selector
                                               + "_" + clf + '_maxd' + str(best_param) + "_split"
                                               + str(copysplit) + "_numfeat" + str(best_numFeat) + '.csv')
                    copyfile(source_best_y, dest_best_y)
                    y_pred_curr = pd.read_csv(dest_best_y)["Survival"].values
                    y_score_curr = pd.read_csv(dest_best_y)["Score"].values

                    y_gt_currsplit = y_gt[copysplit]

                    acc_best_split.append(accuracy_score(y_gt_currsplit, y_pred_curr))
                    acc_balanced_best_split.append(balanced_accuracy_score(y_gt_currsplit, y_pred_curr))
                    f1_best_split.append(f1_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    try:
                        auc_best_split.append(roc_auc_score(y_gt_currsplit, y_score_curr))
                    except:
                        auc_best_split.append(np.NaN)
                    tn, fp, fn, tp = confusion_matrix(y_gt_currsplit, y_pred_curr, labels=[0, 1]).ravel()
                    # sens.append(tp / (tp + fn))
                    sens_best_split.append(recall_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    try:
                        spec_best_split.append(tn / (tn + fp))
                    except:
                        spec_best_split.append(np.NaN)
                    prec_best_split.append(precision_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    # prec.append(tp / (tp + fp))

                auc_overall.loc[selector, clf] = np.nanmean(auc_best_split)
                auc_overall_NAN.loc[selector, clf] = np.isnan(auc_best_split).sum()
                accuracy_overall.loc[selector, clf] = np.nanmean(acc_best_split)
                accuracy_overall_balanced.loc[selector, clf] = np.nanmean(acc_balanced_best_split)
                sensitivity_overall.loc[selector, clf] = np.nanmean(sens_best_split)
                specificity_overall.loc[selector, clf] = np.nanmean(spec_best_split)
                precision_overall.loc[selector, clf] = np.nanmean(prec_best_split)
                f1score_overall.loc[selector, clf] = np.nanmean(f1_best_split)

            elif clf is "Multilayer Perceptron":
                paramspace = [0.0001, 0.001, 0.01, 0.1, 1, 10]
                acclist = np.zeros((len(paramspace), len(numfeatlist)))
                for paramidx, param in enumerate(paramspace):
                    # loop over number of features
                    for nfeatidx, nfeat in enumerate(numfeatlist):
                        acc_split = []
                        for currsplit in range(0, numsplits):
                            y_test_curr_path = os.path.join(twoclass_basepath, str(class_boundary), "ypred_" + selector
                                                            + "_" + clf + '_alpha' + str(param) + "_split"
                                                            + str(currsplit) + "_numfeat" + str(nfeat) + '.csv')

                            y_testpred = pd.read_csv(y_test_curr_path)["Survival"].values
                            acc_split.append(balanced_accuracy_score(y_gt[currsplit], y_testpred))

                        acclist[paramidx, nfeatidx] = np.mean(acc_split)

                i, j = np.unravel_index(acclist.argmax(), acclist.shape)

                best_param = paramspace[i]
                best_numFeat = numfeatlist[j]

                # copy best combination to new folder and calculate metrics
                acc_best_split = []
                auc_best_split = []
                acc_balanced_best_split = []
                sens_best_split = []
                spec_best_split = []
                prec_best_split = []
                f1_best_split = []

                # copy best combination to new folder
                for copysplit in range(0, numsplits):
                    source_best_y = os.path.join(twoclass_basepath, str(class_boundary), "ypred_" + selector
                                                 + "_" + clf + '_alpha' + str(best_param) + "_split"
                                                 + str(copysplit) + "_numfeat" + str(best_numFeat) + '.csv')
                    dest_best_y = os.path.join(twoclass_selected, str(class_boundary), "ypred_" + selector
                                               + "_" + clf + '_alpha' + str(best_param) + "_split"
                                               + str(copysplit) + "_numfeat" + str(best_numFeat) + '.csv')
                    copyfile(source_best_y, dest_best_y)
                    y_pred_curr = pd.read_csv(dest_best_y)["Survival"].values
                    y_score_curr = pd.read_csv(dest_best_y)["Score"].values

                    y_gt_currsplit = y_gt[copysplit]

                    acc_best_split.append(accuracy_score(y_gt_currsplit, y_pred_curr))
                    acc_balanced_best_split.append(balanced_accuracy_score(y_gt_currsplit, y_pred_curr))
                    f1_best_split.append(f1_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    try:
                        auc_best_split.append(roc_auc_score(y_gt_currsplit, y_score_curr))
                    except:
                        auc_best_split.append(np.NaN)
                    tn, fp, fn, tp = confusion_matrix(y_gt_currsplit, y_pred_curr, labels=[0, 1]).ravel()
                    # sens.append(tp / (tp + fn))
                    sens_best_split.append(recall_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    try:
                        spec_best_split.append(tn / (tn + fp))
                    except:
                        spec_best_split.append(np.NaN)
                    prec_best_split.append(precision_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    # prec.append(tp / (tp + fp))

                auc_overall.loc[selector, clf] = np.nanmean(auc_best_split)
                auc_overall_NAN.loc[selector, clf] = np.isnan(auc_best_split).sum()
                accuracy_overall.loc[selector, clf] = np.nanmean(acc_best_split)
                accuracy_overall_balanced.loc[selector, clf] = np.nanmean(acc_balanced_best_split)
                sensitivity_overall.loc[selector, clf] = np.nanmean(sens_best_split)
                specificity_overall.loc[selector, clf] = np.nanmean(spec_best_split)
                precision_overall.loc[selector, clf] = np.nanmean(prec_best_split)
                f1score_overall.loc[selector, clf] = np.nanmean(f1_best_split)

            elif clf is "RBF SVC":
                paramspace1 = [0.25, 0.5, 1, 2, 4]  # costparam
                paramspace2 = ['scale', 'auto', 0.01, 0.1, 1, 10, 100]  # gamma
                acclist = np.zeros((len(paramspace1), len(paramspace2), len(numfeatlist)))
                for paramidx1, param1 in enumerate(paramspace1):
                    # loop over number of features
                    for paramidx2, param2 in enumerate(paramspace2):
                        for nfeatidx, nfeat in enumerate(numfeatlist):
                            acc_split = []
                            for currsplit in range(0, numsplits):
                                y_test_curr_path = os.path.join(twoclass_basepath, str(class_boundary), "ypred_"
                                                                + selector + "_" + clf + '_C' + str(param1) + '_gamma'
                                                                + str(param2) + "_split"
                                                                + str(currsplit) + "_numfeat" + str(nfeat) + '.csv')

                                y_testpred = pd.read_csv(y_test_curr_path)["Survival"].values
                                y_acc = balanced_accuracy_score(y_gt[currsplit], y_testpred)
                                acc_split.append(y_acc)

                            acclist[paramidx1, paramidx2, nfeatidx] = np.mean(acc_split)

                i, j, k = np.unravel_index(acclist.argmax(), acclist.shape)

                best_param1 = paramspace1[i]
                best_param2 = paramspace2[j]
                best_numFeat = numfeatlist[k]

                # copy best combination to new folder and calculate metrics
                acc_best_split = []
                auc_best_split = []
                acc_balanced_best_split = []
                sens_best_split = []
                spec_best_split = []
                prec_best_split = []
                f1_best_split = []
                # copy best combination to new folder
                for copysplit in range(0, numsplits):
                    source_best_y = os.path.join(twoclass_basepath, str(class_boundary), "ypred_" + selector
                                                 + "_" + clf + '_C' + str(best_param1) + '_gamma'
                                                                + str(best_param2) + "_split"
                                                 + str(copysplit) + "_numfeat" + str(best_numFeat) + '.csv')
                    dest_best_y = os.path.join(twoclass_selected,  str(class_boundary), "ypred_" + selector
                                                 + "_" + clf + '_C' + str(best_param1) + '_gamma'
                                                                + str(best_param2) + "_split"
                                                 + str(copysplit) + "_numfeat" + str(best_numFeat) + '.csv')
                    copyfile(source_best_y, dest_best_y)
                    y_pred_curr = pd.read_csv(dest_best_y)["Survival"].values
                    y_score_curr = pd.read_csv(dest_best_y)["Score"].values

                    y_gt_currsplit = y_gt[copysplit]

                    acc_best_split.append(accuracy_score(y_gt_currsplit, y_pred_curr))
                    acc_balanced_best_split.append(balanced_accuracy_score(y_gt_currsplit, y_pred_curr))
                    f1_best_split.append(f1_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    try:
                        auc_best_split.append(roc_auc_score(y_gt_currsplit, y_score_curr))
                    except:
                        auc_best_split.append(np.NaN)
                    tn, fp, fn, tp = confusion_matrix(y_gt_currsplit, y_pred_curr, labels=[0, 1]).ravel()
                    # sens.append(tp / (tp + fn))
                    sens_best_split.append(recall_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    try:
                        spec_best_split.append(tn / (tn + fp))
                    except:
                        spec_best_split.append(np.NaN)
                    prec_best_split.append(precision_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    # prec.append(tp / (tp + fp))

                auc_overall.loc[selector, clf] = np.nanmean(auc_best_split)
                auc_overall_NAN.loc[selector, clf] = np.isnan(auc_best_split).sum()
                accuracy_overall.loc[selector, clf] = np.nanmean(acc_best_split)
                accuracy_overall_balanced.loc[selector, clf] = np.nanmean(acc_balanced_best_split)
                sensitivity_overall.loc[selector, clf] = np.nanmean(sens_best_split)
                specificity_overall.loc[selector, clf] = np.nanmean(spec_best_split)
                precision_overall.loc[selector, clf] = np.nanmean(prec_best_split)
                f1score_overall.loc[selector, clf] = np.nanmean(f1_best_split)

            elif clf is "XGBoost" or clf is "Logistic Regression" or clf is "QDA" or clf is "Naive Bayes" \
                    or clf is "Gaussian Process" or clf is "Random Forest" or clf is "AdaBoost":

                # get best performing auc across number of features
                acclist = np.zeros((len(numfeatlist)))
                for nfeatidx, nfeat in enumerate(numfeatlist):
                    acc_split = []
                    for currsplit in range(0, numsplits):

                        y_test_curr_path = os.path.join(twoclass_basepath, str(class_boundary), "ypred_" + selector
                                                        + "_" + clf
                                                        + "_split" + str(currsplit) + "_numfeat" + str(nfeat) + '.csv')

                        y_testpred = pd.read_csv(y_test_curr_path)["Survival"].values

                        y_acc = balanced_accuracy_score(y_gt[currsplit], y_testpred)
                        acc_split.append(y_acc)

                    acclist[nfeatidx] = np.mean(acc_split)

                k = np.unravel_index(acclist.argmax(), acclist.shape)

                best_numFeat = numfeatlist[k]

                # copy best combination to new folder and calculate metrics
                acc_best_split = []
                auc_best_split = []
                acc_balanced_best_split = []
                sens_best_split = []
                spec_best_split = []
                prec_best_split = []
                f1_best_split = []

                for copysplit in range(0, numsplits):
                    source_best_y = os.path.join(twoclass_basepath, str(class_boundary), "ypred_" + selector
                                                 + "_" + clf + "_split"
                                                 + str(copysplit) + "_numfeat" + str(best_numFeat) + '.csv')
                    dest_best_y = os.path.join(twoclass_selected, str(class_boundary), "ypred_" + selector
                                               + "_" + clf + "_split"
                                               + str(copysplit) + "_numfeat" + str(best_numFeat) + '.csv')
                    copyfile(source_best_y, dest_best_y)
                    y_pred_curr = pd.read_csv(dest_best_y)["Survival"].values
                    y_score_curr = pd.read_csv(dest_best_y)["Score"].values

                    y_gt_currsplit = y_gt[copysplit]

                    acc_best_split.append(accuracy_score(y_gt_currsplit, y_pred_curr))
                    acc_balanced_best_split.append(balanced_accuracy_score(y_gt_currsplit, y_pred_curr))
                    f1_best_split.append(f1_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    try:
                        auc_best_split.append(roc_auc_score(y_gt_currsplit, y_score_curr))
                    except:
                        auc_best_split.append(np.NaN)
                    tn, fp, fn, tp = confusion_matrix(y_gt_currsplit, y_pred_curr, labels=[0, 1]).ravel()
                    # sens.append(tp / (tp + fn))
                    sens_best_split.append(recall_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    try:
                        spec_best_split.append(tn / (tn + fp))
                    except:
                        spec_best_split.append(np.NaN)
                    prec_best_split.append(precision_score(y_gt_currsplit, y_pred_curr, labels=[0, 1]))
                    # prec.append(tp / (tp + fp))

                auc_overall.loc[selector, clf] = np.nanmean(auc_best_split)
                auc_overall_NAN.loc[selector, clf] = np.isnan(auc_best_split).sum()
                accuracy_overall.loc[selector, clf] = np.nanmean(acc_best_split)
                accuracy_overall_balanced.loc[selector, clf] = np.nanmean(acc_balanced_best_split)
                sensitivity_overall.loc[selector, clf] = np.nanmean(sens_best_split)
                specificity_overall.loc[selector, clf] = np.nanmean(spec_best_split)
                precision_overall.loc[selector, clf] = np.nanmean(prec_best_split)
                f1score_overall.loc[selector, clf] = np.nanmean(f1_best_split)


    print("##########")
    print(class_boundary)
    auc_overall.to_csv(os.path.join(aucsavepath, "AUC" + str(class_boundary) + ".csv"))
    auc_overall_NAN.to_csv(os.path.join(aucsavepath, "AUC_NANcount" + str(class_boundary) + ".csv"))
    accuracy_overall.to_csv(os.path.join(aucsavepath, "Accuracy" + str(class_boundary) + ".csv"))
    accuracy_overall_balanced.to_csv(os.path.join(aucsavepath, "Accuracy_balanced" + str(class_boundary) + ".csv"))
    sensitivity_overall.to_csv(os.path.join(aucsavepath, "Sensitivity" + str(class_boundary) + ".csv"))
    specificity_overall.to_csv(os.path.join(aucsavepath, "Specificity" + str(class_boundary) + ".csv"))
    precision_overall.to_csv(os.path.join(aucsavepath, "Precision" + str(class_boundary) + ".csv"))
    f1score_overall.to_csv(os.path.join(aucsavepath, "F1score" + str(class_boundary) + ".csv"))
    print("##########")

