import os
import csv
import glob
import torch

import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.signal import medfilt

class Logger():
    def __init__(self, targetFile, separator=';'):
        self.targetFile = targetFile
        self.separator = separator

    def writeHeaders(self, headers):
        with open(self.targetFile, 'a') as fh:
            for aHeader in headers:
                fh.write(aHeader + self.separator)
            fh.write('\n')

    def writeDataLog(self, dataArray):
        with open(self.targetFile, 'a') as fh:
            for dataItem in dataArray:
                fh.write(str(dataItem) + self.separator)
            fh.write('\n')

def write_to_file(all_data, target):
    with open(target, mode='w') as ef:
        efw = csv.writer(ef, delimiter=',')
        for data in all_data:
            efw.writerow(data)

def prediction_postprocessing(data, filter_lenght):
    positive_predictions = []
    for d in data:
        positive_predictions.append( [float(d[-3]), float(d[-2])] )
    positive_predictions = np.asarray(positive_predictions)

    positive_predictions[..., 0] = medfilt(positive_predictions[..., 0], filter_lenght)
    positive_predictions[..., 1] = medfilt(positive_predictions[..., 1], filter_lenght)
    positive_predictions = softmax(positive_predictions, axis = -1)

    for idx in range(len(data)):
        row = data[idx]
        del row[-2]
        row[-1] = float(positive_predictions[idx][1])
    return data

def select_files(pred_source, gt_source):
    pred_files = glob.glob(pred_source+'/*.csv')
    pred_files.sort()

    gt_files = glob.glob(gt_source+'/*.csv')
    gt_files.sort()

    return pred_files, gt_files

def softmax_feats(source):
    print(source)
    data = csv_to_list(source)

    positive_predictions = []
    for d in data:
        positive_predictions.append( [float(d[-2]), float(d[-1])] )
    positive_predictions = np.asarray(positive_predictions)

    positive_predictions[..., 0] = medfilt(positive_predictions[..., 0], 1)
    positive_predictions[..., 1] = medfilt(positive_predictions[..., 1], 1)
    positive_predictions = softmax(positive_predictions, axis = -1)

    for idx in range(len(data)):
        row = data[idx]
        del row[-2]
        row[-1] = float(positive_predictions[idx][1])
    return data

def load_train_video_set():
    files = os.listdir('/usr/home/kop/active-speakers-context/ava_activespeaker_train_v1.0')
    videos = [f[:-18] for f in files]
    videos.sort()
    return videos

def load_val_video_set():
    files = os.listdir('/usr/home/kop/active-speakers-context/ava_activespeaker_test_v1.0')
    videos = [f[:-18] for f in files]
    videos.sort()
    return videos

def load_test_video_set():
    df = pd.read_csv('/usr/home/kop/active-speakers-context/ava_activespeaker_test_augmented.csv')
    videos = df['video_id'].unique().tolist()
    videos.sort()
    return videos