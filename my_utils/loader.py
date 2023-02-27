import numpy as np
import scipy.io as sio
from os.path import join
from os import listdir, remove
from zipfile import ZipFile
import pandas as pd
from my_utils.gaze import dva2pixels
import re

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)

def load_event_features(file):
    features = np.load(file, allow_pickle=True)
    n_ex = len(features)

    feat_fixs = []
    feat_sacs = []
    stim_fix = []
    stim_sac = []


    for e in range(n_ex):
        curr_data_dict = features[e]
        try:
            feat_fix = curr_data_dict['feat_fix']
            feat_sac = curr_data_dict['sacc_fix']
            
            if feat_fix is None or feat_sac is None:
                continue

            feat_fixs.append(feat_fix)
            stim_fix.append(np.repeat(curr_data_dict['stimulus'], len(feat_fix))[:,np.newaxis])
            feat_sacs.append(feat_sac)
            stim_sac.append(np.repeat(curr_data_dict['stimulus'], len(feat_sac))[:,np.newaxis])
            
        except:
            continue

    feat_fixs = np.vstack(feat_fixs)
    feat_sacs = np.vstack(feat_sacs)
    stim_fix = np.vstack(stim_fix)
    stim_sac = np.vstack(stim_sac)

    return feat_fixs, feat_sacs, stim_fix, stim_sac


def load_reutter(path):
    scanpath = []
    paths = sorted_nicely(listdir(path))
    print(paths)

    for subject in paths:
        sub_scan = load_reutter_sub(path + subject)
        scanpath.append(sub_scan)
    parameters = {
        'distance': 0.53,
        'width': 0.531,
        'height': 0.298,
        'x_res': 1920,
        'y_res': 1080,
        'fs': 1000.}
    scanpath = np.asarray(scanpath)
    return scanpath, parameters


def load_reutter_sub(sub_path):
    # read the whole file into variable `events` (list with one entry per line)
    with open(sub_path) as f:
        events = f.readlines()

    events = [event for event in events if 'SFIX' and 'SSACC' and 'MSG' and 'ESACC' not in event]
    trial_start_indices = np.where(["START" in ev for ev in events])[0]
    trial_end_indices = np.where(["END" in ev for ev in events])[0]

    sub_scan = []
    for i in range(len(trial_start_indices)):  # for each trial
        start = trial_start_indices[i] + 7
        if i != len(trial_start_indices) - 1:
            end = trial_start_indices[i + 1] - 9
        else:
            end = trial_end_indices[-1] - 4

        current_trial = events[ start : end ]  # Removing starting and ending info
        trial_coor = []
        for event in current_trial:
            if 'EFIX' not in event:
                try:
                    x = float(event.split('\t')[1])
                    y = float(event.split('\t')[2])
                    trial_coor.append([x, y])
                except:
                    pass
            else:
                x = float(event.split('\t')[4])  # in case of fixation end x and y coordinates are in positions 4,5
                y = float(event.split('\t')[5])
                trial_coor.append([x, y])

        trial_coor = np.asarray(trial_coor)
        sub_scan.append(trial_coor)
    return np.asarray(sub_scan)


def load_dataset(name, path, round='Round_9', session='S1', task='Video_1'):
    return load_reutter(path)


if __name__ == '__main__':
    data_cerf = load_reutter('../datasets/Reutter')