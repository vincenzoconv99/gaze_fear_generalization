from os.path import join
import scipy
import os
import numpy as np
from my_utils.loader import load_event_features
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
import numpy_indexed as npi
import numpy_indexed as npi
from sklearn.svm import LinearSVR, LinearSVC, OneClassSVM, SVC
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier, EasyEnsembleClassifier, BalancedBaggingClassifier

from sklearn.base import clone
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_approximation import Nystroem
from scipy.stats import uniform
import pandas as pd
import re
from my_utils.plotter import build_roc_curve

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)


def train_sklearn(X, y, model):
    from sklearn.metrics import make_scorer
    
    print('Classification using ', model )

    scorer = make_scorer(f1_score)
    pipe_clf = make_pipeline(RobustScaler(),
                             clone(model)
                             #BalancedRandomForestClassifier()
                             #RUSBoostClassifier(n_estimators = 100, learning_rate=0.001)
                             #RandomForestClassifier(n_estimators = 100, min_samples_split=1000)
                             #Nystroem(gamma=0.002, n_components=1000, kernel='rbf', n_jobs=-1),
                             #LinearSVC(C=1000., max_iter=1000, dual=False)
                             #MLPClassifier(hidden_layer_sizes=(200, 100, 50, 25), max_iter=1000)
                             #SVC(random_state=1, C=1000, gamma=0.002, kernel='rbf')
                            )

    pipe_clf = pipe_clf.fit(X, y)
    y_pred = pipe_clf.predict(X)
    f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    print('\tF1 train: ', f1)
    print('\tAcc train: ', acc)
    return pipe_clf


def evaluate(clf_fix, clf_sac, X_fix_test, y_f_test, stim_f_test, sub_f_test, X_sac_test, y_s_test, stim_s_test, sub_s_test):

    metrics_classification = {}

    #Fixations -------
    ss = np.zeros_like(sub_f_test).astype('str')
    for i in range(len(sub_f_test)):
        ss[i] = str(int(sub_f_test[i])) + '-' + str(int(stim_f_test[i]))
    
    ppred_fix = clf_fix.predict_proba(X_fix_test)

    key_fix, ppred_fix_comb = npi.group_by(ss).mean(ppred_fix)


    #Saccades -------
    ss = np.zeros_like(sub_s_test).astype('str')

    for i in range(len(sub_s_test)):
        ss[i] = str(int(sub_s_test[i])) + '-' + str(int(stim_s_test[i]))

    ppred_sac = clf_sac.predict_proba(X_sac_test)

    key_sac, ppred_sac_comb = npi.group_by(ss).mean(ppred_sac)

    # Extracting labels
    y_test = np.zeros(key_fix.shape)
    for i,k in enumerate(key_fix):
        subject = int(k.split('-')[0])
        stimulus = int(k.split('-')[1])
        y_test[i] = map_ss_shock[(subject, stimulus)]
    
    #Fusion --------
    ppred = np.asarray((np.matrix(ppred_fix_comb) + np.matrix(ppred_sac_comb)) / 2.)
    y_pred = np.squeeze(np.asarray(ppred.argmax(axis=1)))
    y_test = y_test.astype(int)

    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, ppred[:, 1] )
    auprc = average_precision_score(y_test, ppred[:, 1])
    
    print('Classification Evaluation')
    print('\t F1-score test', f1)
    print('\t Accuracy test', accuracy)
    print('\t AUROC test', auroc)
    print('\t AUPRC test', auprc)

    metrics_classification['f1'] = f1
    metrics_classification['accuracy'] = accuracy
    metrics_classification['auroc'] = auroc
    metrics_classification['auprc'] = auprc

    return metrics_classification

def load_dataset(path, nsub=None, num_sessions=None):
    global_data_fix = []
    global_data_sac = []
    subs = sorted_nicely(os.listdir(path))

    behavior_path = './osf/behavior/LookAtMe_'

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

        subject_id = int(file.split("_")[2].split(".")[0])
        behaviour_info = pd.read_csv(behavior_path + '{0:03d}'.format(subject_id) + '.csv', sep='\t')

        shock_list = list(behaviour_info['shock']) # it contains the corresponding shock per trial
        
        fix_shock_labels = [ [ shock_list[x[0]] ] for x in stim_fix]
        sac_shock_labels = [ [ shock_list[x[0]] ] for x in stim_sac]
        
        curr_subject_id_f = np.ones([fix_data.shape[0], 1]) * subject_id
        curr_subject_id_s = np.ones([sac_data.shape[0], 1]) * subject_id

        fix_data = np.hstack([curr_subject_id_f, fix_shock_labels, stim_fix, fix_data])
        sac_data = np.hstack([curr_subject_id_s, sac_shock_labels, stim_sac, sac_data])

        global_data_fix.append(fix_data)
        global_data_sac.append(sac_data)
        subs_considered += 1

    data_fix = np.vstack(global_data_fix)
    data_sac = np.vstack(global_data_sac)
    
    labels_fix = [x[1] for x in data_fix]
    labels_sac = [x[1] for x in data_sac]

    print('Count of 1 in Fixations', labels_fix.count(1))
    print('Count of 0 in Fixations', labels_fix.count(0))
    print('Count of 1 in Saccades', labels_sac.count(1))
    print('Count of 0 in Saccades', labels_sac.count(0))
    
    print('\nLoaded ' + str(subs_considered) + ' subjects...')
    return data_fix, data_sac

def get_CV_splits(stim_f, ids_f, k):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k)
    subs_splits = []
    sub_ids = np.unique(ids_f)
    for s in sub_ids:
        curr_stims = np.unique(stim_f[ids_f==s])[:,np.newaxis]
        subs_splits.append(kf.split(curr_stims))
    return subs_splits, sub_ids

def get_results_kfold(X_fix, ids_f, yf, stim_f, X_sac, ids_s, ys, stim_s, k, model):
    sub_splits_gen, sub_ids = get_CV_splits(stim_f, ids_f, k=k)
    
    sub_splits = {}
    for i,ss in enumerate(sub_splits_gen):
        curr_splits = []
        for train_index, test_index in ss:
            curr_splits.append((train_index, test_index))
        sub_splits[sub_ids[i]] = curr_splits
    
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

        train_ids_f = []
        train_stf = []
        train_ids_s = []
        train_sts = []
        test_ids_f = []
        test_stf = []
        test_ids_s = []
        test_sts = []

        for s in sub_splits.keys():
            curr_Xf = X_fix[ids_f==s,:]
            curr_stf = stim_f[ids_f==s]
            curr_yf = yf[ids_f==s]

            curr_Xs = X_sac[ids_s==s,:]
            curr_sts = stim_s[ids_s==s]
            curr_ys = ys[ids_s==s]            

            train_index = sub_splits[s][fold][0]
            test_index = sub_splits[s][fold][1]
            for ti in train_index:
                train_Xf.append(curr_Xf[curr_stf==ti])
                train_stf.append(curr_stf[curr_stf==ti])
                train_yf.append(curr_yf[curr_stf==ti])
                train_ids_f.append(np.repeat(s, len(train_stf[-1]))) # train subject ids fixations
                
                train_Xs.append(curr_Xs[curr_sts==ti])
                train_sts.append(curr_sts[curr_sts==ti])
                train_ys.append(curr_ys[curr_sts==ti])
                train_ids_s.append(np.repeat(s, len(train_sts[-1]))) # train subject ids saccades

            for ti in test_index:
                test_Xf.append(curr_Xf[curr_stf==ti])
                test_stf.append(curr_stf[curr_stf==ti])
                test_yf.append(curr_yf[curr_stf==ti])
                test_ids_f.append(np.repeat(s, len(test_stf[-1]))) # test subject ids fixations

                test_Xs.append(curr_Xs[curr_sts==ti])
                test_sts.append(curr_sts[curr_sts==ti])
                test_ys.append(curr_ys[curr_sts==ti])
                test_ids_s.append(np.repeat(s, len(test_sts[-1]))) # test subject ids saccades

        train_Xf = np.vstack(train_Xf)
        train_yf = np.concatenate(train_yf)
        train_stf = np.concatenate(train_stf)
        train_ids_f = np.concatenate(train_ids_f)
        train_Xs = np.vstack(train_Xs)
        train_ys = np.concatenate(train_ys)
        train_sts = np.concatenate(train_sts)
        train_ids_s = np.concatenate(train_ids_s)

        test_Xf = np.vstack(test_Xf)
        test_yf = np.concatenate(test_yf)
        test_stf = np.concatenate(test_stf)
        test_ids_f = np.concatenate(test_ids_f)
        test_Xs = np.vstack(test_Xs)
        test_ys = np.concatenate(test_ys)
        test_sts = np.concatenate(test_sts)
        test_ids_s = np.concatenate(test_ids_s)

        print('\nTraining Fixations')
        clf_fix = train_sklearn(train_Xf, train_yf, model=model)
        
        print('Training Saccades')
        clf_sac = train_sklearn(train_Xs, train_ys, model=model)

        current_fold_metrics = evaluate( clf_fix, clf_sac, 
                                         test_Xf, test_yf, test_stf, test_ids_f, 
                                         test_Xs, test_ys, test_sts, test_ids_s)

        cv_metrics.append(current_fold_metrics)
        
    f1s = [fold['f1'] for fold in cv_metrics]
    accuracies = [fold['accuracy'] for fold in cv_metrics]
    aurocs = [fold['auroc'] for fold in cv_metrics]
    auprcs = [fold['auprc'] for fold in cv_metrics]

    return {'f1_score_mean': np.mean(f1s),'f1_score_std': np.std(f1s), 'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies), 'auroc_mean': np.mean(aurocs), 'auroc_std': np.std(aurocs),
            'auprc_mean': np.mean(auprcs), 'auprc_std': np.std(auprcs)}


# MAIN ---------------------------------------------------------------------

dataset_name = 'Reutter_OU_posterior_VI'
models_classification = [ BalancedRandomForestClassifier(), RUSBoostClassifier(n_estimators=100) ]

print('\nReutter Dataset (OU features)...\n')
directory = join(join('features', dataset_name), 'train')

data_fix, data_sac = load_dataset(directory)

map_ss_shock = {} # Mapping subject + stimulus to Shock
for x in data_fix[:, :3]:
    map_ss_shock[(int(x[0]), int(x[2]))] = x[1]

X_fix = data_fix[:, 3:]
ids_f = data_fix[:, 0] # ids dei soggetti (fissazioni)
yf = data_fix[:, 1]
stim_f = data_fix[:, 2] # ids degli stimoli delle fissazioni
X_sac = data_sac[:, 3:]
ids_s = data_sac[:, 0] # ids dei soggetti (saccadi)
ys = data_sac[:, 1]
stim_s = data_sac[:, 2] # ids degli stimoli delle saccadi


n_sub_f = len(np.unique(ids_f))
n_sub_s = len(np.unique(ids_s))
assert n_sub_f == n_sub_s

unique_f, counts_f = np.unique(ids_f, return_counts=True)
cf = dict(zip(unique_f.astype(int), counts_f))

unique_s, counts_s = np.unique(ids_s, return_counts=True)
cs = dict(zip(unique_s.astype(int), counts_s))

print('\n-------------------------------')
print('\nFixations Counts per Subject: \n' + str(cf))
print(' ')
print('Saccades Counts per Subject: \n' + str(cs))
print('\n-------------------------------')

for model in models_classification:

    cv_summary = get_results_kfold( X_fix, ids_f, yf, stim_f, X_sac, ids_s, ys, stim_s, k=5, model=model)

    print('\nF1 CV score: ' + str(cv_summary['f1_score_mean']) + ' +- ' + str(cv_summary['f1_score_std']))
    print('Accuracy CV score: ' + str(cv_summary['accuracy_mean']) + ' +- ' + str(cv_summary['accuracy_std']))
    print('AUROC CV score: ' + str(cv_summary['auroc_mean']) + ' +- ' + str(cv_summary['auroc_std']))
    print('AUPRC CV score: ' + str(cv_summary['auprc_mean']) + ' +- ' + str(cv_summary['auprc_std']))
    print(' ')