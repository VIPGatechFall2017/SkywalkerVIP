import h5py
import numpy as np
import torch
from torch.autograd import Variable

MAPPING = {1:1, 0:0}

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

def data_preprocessing(raw_waveforms, raw_angles, raw_labels):
	new_waveforms, new_angles, new_labels = [], [], []
	for row in range(len(raw_labels)):
	    curr, check = 0, True
	    for col in range(len(raw_labels[0])):
	        if raw_labels[row][col] >= 70:
	            curr += 2**col
	        elif raw_labels[row][col] <= 30:
	            curr = curr
	        else:
	            check = False
	    if check:
	        new_waveforms.append(raw_waveforms[row])
	        new_angles.append(raw_angles[row])
	        new_labels.append(MAPPING[curr])

	new_waveforms, new_angles, new_labels = np.array(new_waveforms), np.array(new_angles), np.array(new_labels)
	train_waveforms = Variable(torch.FloatTensor(new_waveforms).unsqueeze(1))
	train_angles = Variable(torch.FloatTensor(new_angles))
	train_labels = Variable(torch.FloatTensor(new_labels))

	return train_waveforms, train_angles, train_labels

def data_loader(directory):
	raw_waveforms, raw_angles, raw_labels = unpackSingle_h5(directory)
	train_waveforms, train_angles, train_labels = data_preprocessing(raw_waveforms, raw_angles, raw_labels)
	return train_waveforms, train_angles, train_labels
