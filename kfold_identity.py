from os.path import join
import scipy
import os
import numpy as np
import sklearn
from my_utils.loader import load_event_features
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
import numpy_indexed as npi
from sklearn.svm import SVC

from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from scipy.stats import uniform
from sklearn.svm import LinearSVR, LinearSVC, OneClassSVM
from sklearn.kernel_approximation import Nystroem
import pandas as pd
import re
from my_utils.plotter import build_roc_curve
import seaborn as sns

import matplotlib.pyplot as plt
from itertools import cycle

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)


def train_sklearn(X, y, model):
    from sklearn.metrics import make_scorer, f1_score
    scorer = make_scorer(f1_score, average='macro')

    print('Classification using ', model )

    pipe_clf = make_pipeline(RobustScaler(),
                             clone(model) #Creating a new sklearn model with the same parameters of the given one
                            )
    
    pipe_clf = pipe_clf.fit(X, y)
    y_pred = pipe_clf.predict(X)
    f1 = f1_score(y, y_pred, average='weighted')
    acc = accuracy_score(y, y_pred)
    print('\tF1 train: ', f1)
    print('\tAcc train: ', acc)
    return pipe_clf


def evaluate(clf_fix, clf_sac, X_fix_test, y_f_test, stim_f_test, X_sac_test, y_s_test, stim_s_test, fold):
    
    metrics_classification = {}

    #Fixations -------
    ss = np.zeros_like(y_f_test).astype('str')
    for i in range(len(y_f_test)):
        ss[i] = str(int(y_f_test[i])) + '-' + str(int(stim_f_test[i]))

    ppred_fix = clf_fix.predict_proba(X_fix_test)

    key, ppred_fix_comb = npi.group_by(ss).mean(ppred_fix)

    y_test = np.zeros(key.shape) # Mapping orginal subject ids to sequetial identifiers for sklearn compatibility
    d = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 8, 11: 9, 12: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 26: 24, 27: 25, 28: 26, 29: 27, 30: 28, 31: 29, 32: 30, 33: 31, 41: 32, 42: 33, 43: 34, 44: 35, 45: 36, 46: 37, 47: 38, 48: 39, 49: 40, 50: 41, 51: 42, 52: 43, 53: 44, 54: 45, 55: 46}
    for i,k in enumerate(key):
        l = int(k.split('-')[0])
        y_test[i] = d[l]s

    #Saccades -------
    ss = np.zeros_like(y_s_test).astype('str')

    for i in range(len(y_s_test)):
        ss[i] = str(int(y_s_test[i])) + '-' + str(int(stim_s_test[i]))
    
    ppred_sac = clf_sac.predict_proba(X_sac_test)
    
    #Fusion --------
    _, ppred_sac_comb = npi.group_by(ss).mean(ppred_sac)
    ppred = np.asarray((np.matrix(ppred_fix_comb) + np.matrix(ppred_sac_comb)) / 2.)
    y_pred = np.squeeze(np.asarray(ppred.argmax(axis=1)))
    y_test_bin = label_binarize(y_test, classes= np.unique(y_test) )

    #Computing Metrics
    f1 = f1_score(np.array(y_test).astype(int), y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test_bin, ppred, average='weighted')
    auprc = average_precision_score(y_test_bin, ppred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    
    print('Classification Evaluation')
    print('\t F1-score test', f1)
    print('\t Accuracy test', accuracy)
    print('\t AUROC test', auroc)
    print('\t AUPRC test', auprc)


    metrics_classification['f1'] = f1
    metrics_classification['accuracy'] = accuracy
    metrics_classification['auroc'] = auroc
    metrics_classification['auprc'] = auprc
    metrics_classification['cm'] = cm

    return metrics_classification


def load_dataset(path, nsub=None, num_sessions=None):
    global_data_fix = []
    global_data_sac = []
    subs = sorted_nicely(os.listdir(path))
    if nsub is not None:
        subs = subs[:nsub]
    subs_considered = 0
    for file in subs:
        if file == '.DS_Store':
            continue

        fix_data, sac_data, stim_fix, stim_sac = load_event_features(join(path, file))
    
        if num_sessions is not None:
            ns = len(np.unique(stim_fix))
            if ns < num_sessions:
                continue
        label = int(file.split("_")[2].split(".")[0])        
        curr_label_f = np.ones([fix_data.shape[0], 1]) * label
        curr_label_s = np.ones([sac_data.shape[0], 1]) * label
        fix_data = np.hstack([curr_label_f, stim_fix, fix_data])
        sac_data = np.hstack([curr_label_s, stim_sac, sac_data])
        global_data_fix.append(fix_data)
        global_data_sac.append(sac_data)
        subs_considered += 1
    data_fix = np.vstack(global_data_fix)
    data_sac = np.vstack(global_data_sac)
    print('\nLoaded ' + str(subs_considered) + ' subjects...')
    return data_fix, data_sac

def get_CV_splits(stim_f, yf, k):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k)
    subs_splits = []
    sub_labels = np.unique(yf)
    for s in sub_labels:
        curr_stims = np.unique(stim_f[yf==s])[:,np.newaxis]
        subs_splits.append(kf.split(curr_stims))
    return subs_splits, sub_labels

def get_results_kfold(X_fix, yf, stim_f, X_sac, ys, stim_s, k, model):
    sub_splits_gen, sub_labels = get_CV_splits(stim_f, yf, k=k)

    sub_splits = {}
    for i,ss in enumerate(sub_splits_gen):
        curr_splits = []
        for train_index, test_index in ss:
            curr_splits.append((train_index, test_index))
        sub_splits[sub_labels[i]] = curr_splits
    
    cv_metrics = []

    for fold in range(k):
        print('\nFold ' + str(fold+1) + ' of ' + str(k))
        train_Xf = []
        train_yf = []
        train_Xs = []
        train_ys = []
        test_Xf = []
        test_yf = []
        test_Xs = []
        test_ys = []
        train_stf = []
        train_sts = []
        test_stf = []
        test_sts = []
        for s in sub_splits.keys():
            curr_Xf = X_fix[yf==s,:]
            curr_stf = stim_f[yf==s]
            curr_Xs = X_sac[ys==s,:]
            curr_sts = stim_s[ys==s]
            train_index = sub_splits[s][fold][0]
            test_index = sub_splits[s][fold][1]
            for ti in train_index:
                train_Xf.append(curr_Xf[curr_stf==ti])
                train_stf.append(curr_stf[curr_stf==ti])
                train_yf.append(np.repeat(s, len(train_stf[-1])))
                train_Xs.append(curr_Xs[curr_sts==ti])
                train_sts.append(curr_sts[curr_sts==ti])
                train_ys.append(np.repeat(s, len(train_sts[-1])))
            for ti in test_index:
                test_Xf.append(curr_Xf[curr_stf==ti])
                test_stf.append(curr_stf[curr_stf==ti])
                test_yf.append(np.repeat(s, len(test_stf[-1])))
                test_Xs.append(curr_Xs[curr_sts==ti])
                test_sts.append(curr_sts[curr_sts==ti])
                test_ys.append(np.repeat(s, len(test_sts[-1])))
        train_Xf = np.vstack(train_Xf)
        train_yf = np.concatenate(train_yf)
        train_stf = np.concatenate(train_stf)
        train_Xs = np.vstack(train_Xs)
        train_ys = np.concatenate(train_ys)
        train_sts = np.concatenate(train_sts)
        test_Xf = np.vstack(test_Xf)
        test_yf = np.concatenate(test_yf)
        test_stf = np.concatenate(test_stf)
        test_Xs = np.vstack(test_Xs)
        test_ys = np.concatenate(test_ys)
        test_sts = np.concatenate(test_sts)

        print('\nTraining Fixations')
        clf_fix = train_sklearn(train_Xf, train_yf, model=model)
        print('Training Saccades')
        clf_sac = train_sklearn(train_Xs, train_ys, model=model)

        current_fold_metrics = evaluate(clf_fix, clf_sac, test_Xf, test_yf, test_stf, test_Xs, 
                                        test_ys, test_sts, str(fold+1))
        cv_metrics.append(current_fold_metrics)


    f1s = [fold['f1'] for fold in cv_metrics]
    accuracies = [fold['accuracy'] for fold in cv_metrics]
    aurocs = [fold['auroc'] for fold in cv_metrics]
    auprcs = [fold['auprc'] for fold in cv_metrics]
    cms = [fold['cm'] for fold in cv_metrics]

    accumulator = cms[0]
    for i in range(1, len(cms)):
        accumulator += np.array(cms[i])

    #Saving global onfusion matrix of the model
    sns.heatmap(accumulator, annot=False, fmt='.2%', cmap='Blues')
    plt.savefig('./images/identity_cm_' + str(model) + '.png') 

    #Returning means and stds of the metrics
    return {'f1_score_mean': np.mean(f1s),'f1_score_std': np.std(f1s), 'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies), 'auroc_mean': np.mean(aurocs), 'auroc_std': np.std(aurocs),
            'auprc_mean': np.mean(auprcs), 'auprc_std': np.std(auprcs)}


# MAIN ---------------------------------------------------------------------


dataset_name = 'Reutter_OU_posterior_VI'

models_classification = [ SVC(random_state=1, C=1000, gamma=0.002, kernel='rbf', probability=True),
                          RandomForestClassifier(min_samples_split=10),
                          MLPClassifier(hidden_layer_sizes=(100, 75, 50)) ]

print('\nReutter Dataset (OU features)...\n')
directory = join(join('features', dataset_name), 'train')
data_fix, data_sac = load_dataset(directory)


X_fix = data_fix[:, 2:]
yf = data_fix[:, 0] # Subjects' ids (fixations)
stim_f = data_fix[:, 1] # Stimulus' ids (fixations)
X_sac = data_sac[:, 2:]
ys = data_sac[:, 0] # Subjects' ids (saccades)
stim_s = data_sac[:, 1] # Stimulus' ids (saccades)


n_class_f = len(np.unique(yf))
n_class_s = len(np.unique(ys))
assert n_class_f == n_class_s

print('\nNumber of classes: ' + str(n_class_f))

unique_f, counts_f = np.unique(yf, return_counts=True)
cf = dict(zip(unique_f.astype(int), counts_f))

unique_s, counts_s = np.unique(ys, return_counts=True)
cs = dict(zip(unique_s.astype(int), counts_s))

print('\n-------------------------------')
print('\nFixations Counts per Class: \n' + str(cf))
print(' ')
print('Saccades Counts per Class: \n' + str(cs))
print('\n-------------------------------')

for model in models_classification:

    cv_summary = get_results_kfold( X_fix, yf, stim_f, X_sac, ys, stim_s, k=5, model=model)

    print('\nF1 CV score: ' + str(cv_summary['f1_score_mean']) + ' +- ' + str(cv_summary['f1_score_std']))
    print('Accuracy CV score: ' + str(cv_summary['accuracy_mean']) + ' +- ' + str(cv_summary['accuracy_std']))
    print('AUROC CV score: ' + str(cv_summary['auroc_mean']) + ' +- ' + str(cv_summary['auroc_std']))
    print('AUPRC CV score: ' + str(cv_summary['auprc_mean']) + ' +- ' + str(cv_summary['auprc_std']))
    print(' ')