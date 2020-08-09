#!/usr/bin/env python3

import os
import pandas as pd
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
# from skfeature.function import similarity_based, information_theoretical_based, statistical_based

from skfeature.function.similarity_based import reliefF, fisher_score
from skfeature.function.statistical_based import chi_square, t_score, gini_index
from skfeature.function.information_theoretical_based import CIFE, JMI, DISR, MIM, CMIM, ICAP, MRMR, MIFS

from sklearn.metrics import roc_auc_score

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


def survival_classencoding(survarr: np.array, classboundaries: list):
    if len(classboundaries) == 1:
        survival_classes = [0 if elem <= classboundaries[0] else 1 for elem in survarr]

    if len(classboundaries) == 2:
        survival_classes = [0 if elem <= classboundaries[0] else 1 if elem <= classboundaries[1] else 2 for elem in
                            survarr]

    return survival_classes


def main(input_features_path_csv: str, output_path: str, survival_column: str, exclude_columns: str,
         class_boundary_list: list, num_folds: int, split_path: str, save_splits: bool):
    print(args.input_features_path_csv)
    print(input_features_path_csv)

    np.random.seed(42)

    assert (os.path.isfile(input_features_path_csv)), "Input feature csv file "

    assert (type(num_folds) is int and num_folds > 1), "Please enter an int > 1 for the number of folds."

    input_features = pd.read_csv(input_features_path_csv)

    assert (survival_column in input_features.columns), "Survival column not found"

    numfeatlist = np.arange(5, 55, 5)
    # numfeatlist = np.arange(5, 30, 5)

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

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(n_estimators=100, max_features='auto'),
        MLPClassifier(alpha=1, max_iter=5000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        XGBClassifier,
        LogisticRegression()
    ]

    selectors = [
        reliefF.reliefF,
        fisher_score.fisher_score,
        gini_index.gini_index,
        chi_square.chi_square,
        JMI.jmi,
        CIFE.cife,
        DISR.disr,
        MIM.mim,
        CMIM.cmim,
        ICAP.icap,
        t_score.t_score,
        MRMR.mrmr,
        MIFS.mifs
    ]

    selectornames = ["Relief", "Fisher score", "Gini index", "Chi-square", "Joint mutual information",
                     "Conditional infomax feature extraction", "Double input symmetric relevance",
                     "Mutual information maximization", "Conditional mutual information maximization",
                     "Interaction capping", "T-test", "Minimum redundancy maximum relevance",
                     "Mutual information feature selection"]
    selectornames_short = ["RELF", "FSCR", "GINI", "CHSQ", "JMI", "CIFE", "DISR", "MIM", "CMIM", "ICAP", "TSCR", "MRMR",
                           "MIFS"]

    for class_boundary in class_boundary_list:
        currsurvenc = input_features
        y_arr = pd.Series(survival_classencoding(input_features[survival_column].values, [class_boundary])).values
        y = pd.DataFrame(survival_classencoding(input_features[survival_column].values, [class_boundary]),
                         columns=["Survival"])
        # y = currsurvenc[survival_column].values
        X = currsurvenc.drop(survival_column, axis=1, inplace=False)
        try:
            X.drop("ID", axis=1, inplace=True)
        except:
            print('No ID column found')

        X_arr = X.values

        # generate splits for stratified cross validation
        skf = StratifiedKFold(n_splits=num_folds, random_state=None, shuffle=False)
        splitcount = 0
        numfeat = np.linspace(5, 50, 10)
        for train_index, test_index in skf.split(X_arr, y_arr):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X_arr[train_index], X_arr[test_index]
            y_train, y_test = y_arr[train_index], y_arr[test_index]

            X_train_df = X.iloc[train_index]
            X_test_df = X.iloc[test_index]
            y_train_df = y.iloc[train_index]
            y_test_df = y.iloc[test_index]

            X_train_df.to_csv(os.path.join(output_path, 'Xtrain_2class_split_boundary' + str(class_boundary) + '_split'
                                           + str(splitcount) + '.csv'), index=False)
            X_test_df.to_csv(os.path.join(output_path, 'Xtest_2class_split_boundary' + str(class_boundary) + '_split'
                                          + str(splitcount) + '.csv'), index=False)
            y_train_df.to_csv(os.path.join(output_path, 'ytrain_2class_split_boundary' + str(class_boundary) + '_split'
                                           + str(splitcount) + '.csv'), index=False)
            y_test_df.to_csv(os.path.join(output_path, 'ytest_2class_split_boundary' + str(class_boundary) + '_split'
                                          + str(splitcount) + '.csv'), index=False)

            # if it does not already exist, create folder for each class boundary to store selected features and results
            boundarydir = os.path.join(output_path, str(class_boundary))
            if not os.path.exists(boundarydir):
                os.makedirs(boundarydir)

            # for every split, perform feature selection
            for sel_name, sel in zip(selectornames_short, selectors):
                print('#####')
                print(sel_name)
                print('#####')

                if sel_name is "CHSQ":
                    # shift X values to be non-negative for chsq feature selection
                    X_train_tmp = X_train + np.abs(X_train.min())
                    selscore = sel(X_train_tmp, y_train)
                    selidx = np.argsort(selscore)[::-1]
                    selidx = selidx[0:50]
                    selscore = selscore[selidx]
                    selscoredf = pd.DataFrame(
                        data=np.transpose(np.vstack((X_train_df.columns[selidx].values, selscore))),
                        columns=['Feature', 'Score'])
                    selscoredf.to_csv(
                        os.path.join(boundarydir, sel_name + '_50features_split' + str(splitcount) + '.csv'),
                        index=None)

                elif sel_name == "RELF":
                    selscore = sel(X_train, y_train, k=50)

                    selidx = np.argsort(selscore)[::-1]
                    # print(selidx)
                    selidx = selidx[0:50]
                    selscoredf = pd.DataFrame(
                        data=np.transpose(np.vstack((X_train_df.columns[selidx].values, selscore[selidx]))),
                        columns=['Feature', 'Score'])
                    selscoredf.to_csv(
                        os.path.join(boundarydir, sel_name + '_50features_split' + str(splitcount) + '.csv'),
                        index=None)

                elif sel_name == "JMI" or sel_name == "CIFE" or sel_name == "DISR" or sel_name == "MIM" \
                        or sel_name == "CMIM" or sel_name == "ICAP" or sel_name == "MRMR" or sel_name == "MIFS":
                    selidx, selscore, _ = sel(X_train, y_train, n_selected_features=50)
                    selscoredf = pd.DataFrame(
                        data=np.transpose(np.vstack((X_train_df.columns[selidx].values, selscore))),
                        columns=['Feature', 'Score'])
                    selscoredf.to_csv(
                        os.path.join(boundarydir, sel_name + '_50features_split' + str(splitcount) + '.csv'),
                        index=None)

                else:
                    selscore = sel(X_train, y_train)

                    selidx = np.argsort(selscore)[::-1]
                    # print(selidx)
                    selidx = selidx[0:50]
                    selscoredf = pd.DataFrame(
                        data=np.transpose(np.vstack((X_train_df.columns[selidx].values, selscore[selidx]))),
                        columns=['Feature', 'Score'])
                    selscoredf.to_csv(
                        os.path.join(boundarydir, sel_name + '_50features_split' + str(splitcount) + '.csv'),
                        index=None)

                # get subsets for all number of features
                for numfeat in numfeatlist:
                    X_train_selected = X_train[:, selidx[0:numfeat]]
                    X_test_selected = X_test[:, selidx[0:numfeat]]

                    ##########################################
                    # do classification with all classifiers #
                    ##########################################
                    for clf_name, clf in zip(classifiernames, classifiers):
                        print(clf_name)
                        if clf_name is "XGBoost":
                            clf = XGBClassifier()
                            clf.fit(X_train_selected, y_train)
                            # score = clf.score(X_test_selected, y_test)
                            if hasattr(clf, "decision_function"):
                                score = clf.decision_function(X_test_selected)[:, 1]
                            else:
                                score = clf.predict_proba(X_test_selected)[:, 1]
                            y_pred = clf.predict(X_test_selected)
                            y_train_pred = clf.predict(X_train_selected)

                            # auc = roc_auc_score(y_test, y_pred)
                            # print('Number of features: ' + str(numfeat) + ', ' + name + ': ' + str(auc))
                            pd.DataFrame(data=np.transpose(np.vstack((y_pred, score))), columns=['Survival', 'Score'])\
                                .to_csv(os.path.join(boundarydir, 'ypred_' + sel_name + '_' + clf_name + '_split'
                                                     + str(splitcount) + '_numfeat' + str(numfeat)
                                                     + '.csv'), index=None)
                            pd.DataFrame(data=y_train_pred, columns=['Survival'])\
                                .to_csv(os.path.join(boundarydir, 'ytrainpred_' + sel_name + '_' + clf_name + '_split'
                                                     + str(splitcount) + '_numfeat' + str(numfeat)
                                                     + '.csv'), index=None)

                        elif clf_name is "Nearest Neighbors":
                            numneighbors = np.arange(3, 22, 3)
                            for num_n in numneighbors:
                                clf = KNeighborsClassifier(n_neighbors=num_n)
                                clf.fit(X_train_selected, y_train)
                                # score = clf.score(X_test_selected, y_test)
                                if hasattr(clf, "decision_function"):
                                    score = clf.decision_function(X_test_selected)[:, 1]
                                else:
                                    score = clf.predict_proba(X_test_selected)[:, 1]
                                y_pred = clf.predict(X_test_selected)
                                y_train_pred = clf.predict(X_train_selected)

                                # auc = roc_auc_score(y_test, y_pred)
                                # print('Number of features: ' + str(numfeat) + ', ' + name + ': ' + str(auc))
                                pd.DataFrame(data=np.transpose(np.vstack((y_pred, score))),
                                             columns=['Survival', 'Score']) \
                                    .to_csv(os.path.join(boundarydir, 'ypred_' + sel_name + '_' + clf_name + '_numN'
                                                         + str(num_n) + '_split'
                                                         + str(splitcount) + '_numfeat' + str(numfeat)
                                                         + '.csv'), index=None)
                                pd.DataFrame(data=y_train_pred,
                                             columns=['Survival']) \
                                    .to_csv(os.path.join(boundarydir, 'ytrainpred_' + sel_name + '_' + clf_name + '_numN'
                                                         + str(num_n) + '_split'
                                                         + str(splitcount) + '_numfeat' + str(numfeat)
                                                         + '.csv'), index=None)

                        elif clf_name is "Linear SVC":
                            costparam = [0.25, 0.5, 1, 2, 4]
                            for c in costparam:
                                clf = SVC(kernel="linear", C=c)
                                clf.fit(X_train_selected, y_train)
                                # score = clf.score(X_test_selected, y_test)
                                if hasattr(clf, "decision_function"):
                                    score = clf.decision_function(X_test_selected)
                                else:
                                    score = clf.predict_proba(X_test_selected)[:, 1]
                                y_pred = clf.predict(X_test_selected)
                                y_train_pred = clf.predict(X_train_selected)

                                pd.DataFrame(data=np.transpose(np.vstack((y_pred, score))),
                                             columns=['Survival', 'Score']) \
                                    .to_csv(os.path.join(boundarydir, 'ypred_' + sel_name + '_' + clf_name + '_C'
                                                         + str(c) + '_split'
                                                         + str(splitcount) + '_numfeat' + str(numfeat)
                                                         + '.csv'), index=None)
                                pd.DataFrame(data=y_train_pred,
                                             columns=['Survival']) \
                                    .to_csv(os.path.join(boundarydir, 'ytrainpred_' + sel_name + '_' + clf_name + '_C'
                                                         + str(c) + '_split'
                                                         + str(splitcount) + '_numfeat' + str(numfeat)
                                                         + '.csv'), index=None)

                        elif clf_name is "RBF SVC":
                            costparam = [0.25, 0.5, 1, 2, 4]
                            gamma = ['scale', 'auto', 0.01, 0.1, 1, 10, 100]
                            for c in costparam:
                                for g in gamma:
                                    clf = SVC(gamma=g, C=c)
                                    clf.fit(X_train_selected, y_train)
                                    # score = clf.score(X_test_selected, y_test)
                                    if hasattr(clf, "decision_function"):
                                        score = clf.decision_function(X_test_selected)
                                    else:
                                        score = clf.predict_proba(X_test_selected)[:, 1]
                                    y_pred = clf.predict(X_test_selected)
                                    y_train_pred = clf.predict(X_train_selected)

                                    pd.DataFrame(data=np.transpose(np.vstack((y_pred, score))),
                                                 columns=['Survival', 'Score']) \
                                        .to_csv(os.path.join(boundarydir, 'ypred_' + sel_name + '_' + clf_name + '_C'
                                                             + str(c) + '_gamma' + str(g) + '_split'
                                                             + str(splitcount) + '_numfeat' + str(numfeat)
                                                             + '.csv'), index=None)
                                    pd.DataFrame(data=y_train_pred,
                                                 columns=['Survival']) \
                                        .to_csv(os.path.join(boundarydir, 'ytrainpred_' + sel_name + '_' + clf_name + '_C'
                                                             + str(c) + '_gamma' + str(g) + '_split'
                                                             + str(splitcount) + '_numfeat' + str(numfeat)
                                                             + '.csv'), index=None)

                        elif clf_name is "Decision Tree":
                            maxdepthlist = [5, 10, 15, 20]
                            for d in maxdepthlist:
                                clf = DecisionTreeClassifier(max_depth=d)
                                clf.fit(X_train_selected, y_train)
                                # score = clf.score(X_test_selected, y_test)
                                if hasattr(clf, "decision_function"):
                                    score = clf.decision_function(X_test_selected)[:, 1]
                                else:
                                    score = clf.predict_proba(X_test_selected)[:, 1]
                                y_pred = clf.predict(X_test_selected)
                                y_train_pred = clf.predict(X_train_selected)

                                # auc = roc_auc_score(y_test, y_pred)
                                # print('Number of features: ' + str(numfeat) + ', ' + name + ': ' + str(auc))
                                pd.DataFrame(data=np.transpose(np.vstack((y_pred, score))),
                                             columns=['Survival', 'Score']) \
                                    .to_csv(os.path.join(boundarydir, 'ypred_' + sel_name + '_' + clf_name + '_maxd'
                                                         + str(d) + '_split'
                                                         + str(splitcount) + '_numfeat' + str(numfeat)
                                                         + '.csv'), index=None)
                                pd.DataFrame(data=y_train_pred,
                                             columns=['Survival']) \
                                    .to_csv(os.path.join(boundarydir, 'ytrainpred_' + sel_name + '_' + clf_name + '_maxd'
                                                         + str(d) + '_split'
                                                         + str(splitcount) + '_numfeat' + str(numfeat)
                                                         + '.csv'), index=None)

                        elif clf_name is "Multilayer Perceptron":
                            alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10]
                            for a in alpha:
                                clf = MLPClassifier(alpha=a, max_iter=5000)
                                clf.fit(X_train_selected, y_train)
                                # score = clf.score(X_test_selected, y_test)
                                if hasattr(clf, "decision_function"):
                                    score = clf.decision_function(X_test_selected)[:, 1]
                                else:
                                    score = clf.predict_proba(X_test_selected)[:, 1]
                                y_pred = clf.predict(X_test_selected)
                                y_train_pred = clf.predict(X_train_selected)

                                pd.DataFrame(data=np.transpose(np.vstack((y_pred, score))),
                                             columns=['Survival', 'Score']) \
                                    .to_csv(os.path.join(boundarydir, 'ypred_' + sel_name + '_' + clf_name + '_alpha'
                                                         + str(a) + '_split'
                                                         + str(splitcount) + '_numfeat' + str(numfeat)
                                                         + '.csv'), index=None)
                                pd.DataFrame(data=y_train_pred,
                                             columns=['Survival']) \
                                    .to_csv(os.path.join(boundarydir, 'ytrainpred_' + sel_name + '_' + clf_name + '_alpha'
                                                         + str(a) + '_split'
                                                         + str(splitcount) + '_numfeat' + str(numfeat)
                                                         + '.csv'), index=None)

                        else:
                            clf.fit(X_train_selected, y_train)
                            # score = clf.score(X_test_selected, y_test)
                            if hasattr(clf, "decision_function"):
                                score = clf.decision_function(X_test_selected)
                                # print(score)
                            else:
                                score = clf.predict_proba(X_test_selected)[:, 1]
                            y_pred = clf.predict(X_test_selected)
                            y_train_pred = clf.predict(X_train_selected)

                            # auc = roc_auc_score(y_test, y_pred)
                            # print('Number of features: ' + str(numfeat) + ', ' + name + ': ' + str(auc))
                            pd.DataFrame(data=np.transpose(np.vstack((y_pred, score))), columns=['Survival', 'Score'])\
                                .to_csv(os.path.join(boundarydir, 'ypred_' + sel_name + '_' + clf_name
                                                     + '_split' + str(splitcount) + '_numfeat' +str(numfeat)
                                                     + '.csv'), index=None)
                            pd.DataFrame(data=y_train_pred, columns=['Survival'])\
                                .to_csv(os.path.join(boundarydir, 'ytrainpred_' + sel_name + '_' + clf_name
                                                     + '_split' + str(splitcount) + '_numfeat' +str(numfeat)
                                                     + '.csv'), index=None)

            # print(splitcount)
            splitcount += 1


if __name__ == "__main__":
    """The program's entry point."""
    parser = argparse.ArgumentParser(description='Run stratified k-fold cross-validation for feature selection. '
                                                 'Version for two classes')

    parser.add_argument(
        '--input_features_path_csv',
        type=str,
        help='Path to the csv file containing both features and class labels.'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        help='Path where the csv with selected features will be stored. Will be created, if it does not exist.'
    )

    parser.add_argument(
        '--survival_column',
        type=str,
        default="Survival",
        help='Column name in the file input_features_path_csv containing the survival information. Use the same units '
             'as in class_boundary_list (e.g. days).'
    )

    parser.add_argument(
        '--exclude_columns',
        type=list,
        default=[],
        help='If input_features_path_csv columns contains non-feature columns (e.g. patient IDs), list here to be '
             'excluded. These columns will be dropped and do not appear in the feature ranking.'
    )

    parser.add_argument(
        '--class_boundary_list',
        type=float,
        default=[240],  #, [240, 304.2, 365, 425.8, 540, 730]
        # default=[365],
        help='Class boundaries to create classes from continous input. Values <= boundary will be assigned to class 0, '
             '> boundary to class 1'
    )

    parser.add_argument(
        '--num_folds',
        type=int,
        default=10,
        help='Number of folds for stratified cross-validation.'
    )

    parser.add_argument(
        '--split_path',
        type=str,
        help='Path to splits, either for loading or saving.'
    )

    parser.add_argument(
        '--save_splits',
        type=bool,
        default=True,
        help='To save the splits, set to True. Please note that the columns listed in exclude_columns are not dropped '
             'in the saved splits, but the continous input encoded in classes according to class_boundary_list. A '
             'split file is saved per entry in class_boundary_list'
    )

    args = parser.parse_args()

main(args.input_features_path_csv, args.output_path, args.survival_column, args.exclude_columns,
     args.class_boundary_list, args.num_folds, args.split_path, args.save_splits)

