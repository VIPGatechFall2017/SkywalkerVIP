''' Some of the code are modified based on Hanoi's code'''

import h5py
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pylab as plt
from collections import Counter
from sklearn.utils import shuffle

MAPPING = {29:1, 25:2, 24:3, 1:4, 0:5, 17:6, 9:7, 5:8, 3:9}

'''
Functions for loading raw data and raw labels -> The problem with these functions is how they
load EVERYTHING into memory, and try to process it. This causes memeory problems even on a
32 GB RAM system
'''
# Uses with Hanoi's naming convention (differentiates angles)
def unpackSingle_h5(directory):
    #print("loading data from " + directory)
    hf = h5py.File(directory + '/ultrasound_data.h5', 'r')
    ultrasound_waveforms = hf['ultrasound'][:]
    angles = hf['angles'][:]
    labels = hf['labels'][:]
    return ultrasound_waveforms, angles, labels

def data_preprocessing(raw_waveforms, raw_angles, raw_labels, train=False, upbound=60, downbound=40, balance=False):
    new_waveforms, new_angles, new_labels = [], [], []
    if balance:
        _wave, _angles, _labels, count = [], [], [], dict()
        for row in range(len(raw_labels)):
            curr = 0
            for col in range(len(raw_labels[0])):
                if raw_labels[row][col] > upbound:
                    curr += 2**col
                elif raw_labels[row][col] <= downbound:
                    curr = curr
            if train:
                if curr in MAPPING.keys():
                    new_waveforms.append(raw_waveforms[row])
                    new_angles.append(raw_angles[row])
                    new_labels.append(MAPPING[curr])
        dist = Counter(new_labels)
        limit = np.median(dist.values())
        for k in dist.keys():
            count[k] = 0
        for i in range(0, len(new_labels)):
            if count[new_labels[i]] < limit:
                _wave.append(new_waveforms[i])
                _labels.append(new_labels[i])
                _angles.append(new_angles[i])
                count[new_labels[i]] += 1
        _, _labels = shuffle(_wave, _labels, random_state=0)
        _wave, _angles = shuffle(_wave, _angles, random_state=0)
        return np.array(_wave), np.array(_angles), np.array(_labels)
    else:
        for row in range(len(raw_labels)):
            curr = 0
            for col in range(len(raw_labels[0])):
                if raw_labels[row][col] > upbound:
                    curr += 2**col
                elif raw_labels[row][col] <= downbound:
                    curr = curr
            if curr in MAPPING.keys():
                new_waveforms.append(raw_waveforms[row])
                new_angles.append(raw_angles[row])
                new_labels.append(MAPPING[curr])
        return np.array(new_waveforms), np.array(new_angles), np.array(new_labels)

def to_var(x, types='Float'):
    if types == 'Float':
        return Variable(torch.FloatTensor(x))
    else:
        return Variable(torch.LongTensor(x))

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')