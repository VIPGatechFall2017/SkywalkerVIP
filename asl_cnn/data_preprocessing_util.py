'''
`data_preprocessing_util.py` - Lamtharn Hanoi Hantrakul

Contains functions for preprocessing data. These are called by both `Model()` classes defined in `../model_defs` as well
as `data_visualizer.py` and `model_visualizer.py`
'''

import numpy as np
import scipy
from skimage.measure import block_reduce
from scipy.signal import hilbert, chirp, medfilt, decimate, butter, lfilter
from scipy import ndimage

'''
########## NOTE on FUNCTION PARAMATERS & SIGNATURES ###########
In all of these functions:
:param `data` (numpy matrix) 
    Input data matrix either has dimensions (n,5,2080) or (5,2080). Thus, every preprocessing function
    has a check for these number of dimensions. This is so that the same function can be used during training
    large datasets (n,5,2080) and inference on individual (5,2080) real-time samples. 

:param `verbose` (bool)
    Determines whether you are printing out the name of the preprocessing function
    
:return `modified_data` (numpy matrix)
    The numpy matrix but modified using the preprocessing function

'''

'''
Functions for Normalization
'''
def normalize(data, verbose=True):
    '''
    Normalize entire 5x2080 matrix by the max value
    '''
    if verbose:
        print("++ Normalizing data")

    if data.ndim == 2:
        # if data comes in as a single data point (5x2080),
        return data / np.max(np.abs(data))
    else:
        # data came in as a 3 dimension array (20000x5x2080),
        # normalizing is a little more involved in this case
        # You could trivially normalize sample by sample, but you can vectorize code as matrix multiplication
        (N,num_elems,num_timesteps) = data.shape
        reshaped_data = np.reshape(data, (N,-1))  # reshape to (20000x10400)
        max_vals= np.max(np.abs(reshaped_data), axis=1) # `max_vals` is a (20000x1) matrix holding denominator of normalization
        norm_data = (reshaped_data.T * 1/max_vals)  # rephrase division by `max_val` as multiplication by `hanoi_rapid_thumb_midi_1/max_val` (faster)
        norm_data = norm_data.T # change (10400x20000) from previous step back to (20000x10400)
        reshaped_norm_data = np.reshape(norm_data,(N,num_elems,num_timesteps)) # reshape back to original dimensions
        return reshaped_norm_data

def normalizePerChannel(data, verbose=True):
    '''
    Normalize per channel (i.e 1x2080) in full matrix 5x0280
    '''
    if verbose:
        print("++ Normalizing data Per Channel")
    if data.ndim == 2:
        # if data comes in as a single data point (5x2080),
        return data/(np.max(np.abs(data), axis=1)[:,None])
    else:
        return data/(np.max(np.abs(data), axis=2)[:,:,None])

'''
Functions for dividing all values by a certain factor
'''
def divideByFactor(data, factor=100, verbose=True):
    if verbose:
        print("++ Dividing data by constant factor:", factor)
    return data / factor

'''
Functions for Downsampling
'''
def downsample(data, factor=15, verbose=True):
    if verbose:
        print("++ Downsampling data")

    if data.ndim == 2:
        # if data comes in as a single data point (5x2080),
        return decimate(data,factor,axis=1,zero_phase=False)
    else:
        # data came in as an hanoi_rapid_thumb_midi_3 dimension array (20000x5x2080),
        return decimate(data,factor,axis=2,zero_phase=False)
'''
Functions for flattening data
'''
def flatten(data, verbose=True):
    if verbose:
        print("++ Flattening data")
    if data.ndim == 2:
        # if data comes in as a single data point (5x2080),
        return data.flatten()
    else:
        # if data comes in as a matrix (20000x5x2080),
        return np.reshape(data,(data.shape[0],-1))

'''
Functions for finding centroid of data
'''

def extractCentroid(data, verbose=True, plotting=False):
    if verbose:
        print("++ Extracting Centroid")

    if data.ndim == 2:
        # if data comes in as a single data point (5x2080),
        mask_matrix = (np.arange(0, data.shape[1]))
        mask_matrix = np.expand_dims(mask_matrix, axis=0)
        mask_matrix = np.repeat(mask_matrix, data.shape[0], axis=0) # `mask_matrix` is a (5,2080) matrix with each row being [0 1 2 3 4 ... 2078 2079]
        numer = np.multiply(data, mask_matrix)
        numer = np.sum(numer,axis=1)

        denom = np.sum(data,axis=1)
        centroid = np.multiply(numer, 1/denom)
        centroid = np.expand_dims(centroid, axis=1) # make this (5,1)

        if (plotting):  # make the returned data compatible with `visualize_model.py`
            centroid = formatCentroidForPlot(centroid, data.shape)

        return centroid
    else:
        # data came in as a large dataset (20000x5x2080)
        mask_matrix = (np.arange(0, data.shape[2]))
        mask_matrix = np.expand_dims(mask_matrix, axis=0)
        mask_matrix = np.repeat(mask_matrix, data.shape[1], axis=0)
        mask_matrix = np.expand_dims(mask_matrix, axis=0) # make `mask_matrix` (1,5,2080)
        numer = np.multiply(data, mask_matrix)
        numer = np.sum(numer, axis=2)

        denom = np.sum(data, axis=2)
        centroid = np.multiply(numer, 1 / denom)
        centroid = np.expand_dims(centroid, axis=2) # make this (20000,5,1)
        return centroid

def formatCentroidForPlot(centroid, data_dim=(5,2080)):
    '''
    This function makes plotting the centrolid in `visualize_data.py` possible
    :param centroid: (5x1) or (nx5x1) matrix containing the centroid of the ultrasound echoes
    :return: 5x2080 matrix where most entries are `NaN` except for a bli[ corresponding to the centroid
    '''
    centroid_plot = np.full(data_dim, 0)
    for i in range(5):
        centroid_plot[i,int(centroid[i])] = 1.0  # make an impulse blip at the centroid location
    return centroid_plot
'''
Functions for Median Filtering
'''
# I tried using this one and the result looks a little weird
def medianFilter(data, kernel=3):
    return medfilt(data,kernel)

# This is adapted from an image processing library, except you run it in 1D and not 2D
def medianFilter2(data, size=3, verbose=True):
    if verbose:
        print("++ Median Filtering data")
    if data.ndim == 2:
        return block_reduce(data, block_size=(1, size), func=np.mean)
    else:
        return block_reduce(data, block_size=(1, 1, size), func=np.mean)
'''
Functions for Trimming
'''
def trim(data, start_idx=10, verbose=True):
    if verbose:
        print("++ Trimming data")

    if data.ndim == 2:
        # if data comes in as a single data point (5x2080),
        return data[:,start_idx:]
    else:
        # data came in as an hanoi_rapid_thumb_midi_3 dimension array (20000x5x2080),
        return data[:,:,start_idx:]


'''
Functions for changing sampled depth

At 40MS/s the samples are taken at 25 ns intervals.
So the 5x2080 matrix returned is 2080*25 = 52 microseconds worth of data

This means we can calculate depth "seen" by this matrix of data
d = (1500 m/s) * (52 us) / (2) = 0.04 m = 4cm

If we want to get 3 cm of "vision", then we can truncate the matrix further
'''
def returnDepth(data, depth, discard_start=0, fs=40e6):
    pass

'''
Functions for  bandpass filtering
'''
def bandpassFilter(data, f_low=2e6, f_high=8e6, verbose=True):
    if verbose:
        print("++ Bandpassing data")

    #This function is by Bernie Shih
    fs = 40e6
    nyq = fs / 2
    flow = f_low / nyq
    fhigh = f_high / nyq
    order = 5

    # bandpass-pass filter coefficients
    b, a = scipy.signal.butter(order, [flow, fhigh], btype='bandpass')

    if data.ndim == 2:
        # if data comes in as a single data point (5x2080),
        return scipy.signal.lfilter(b, a, data, axis=1)
    else:
        # data came in as an hanoi_rapid_thumb_midi_3 dimension array (20000x5x2080),
        return scipy.signal.lfilter(b, a, data, axis=2)

'''
Functions for Envelope detection
Note that for large datasets envelope detection is a very expensive/time consuming operation
'''
def envelope(data, verbose=True):
    if verbose:
        print("++ Enveloping data")

    if data.ndim == 2:
        # if data comes in as a single data point (5x2080),
        return np.abs(hilbert(data, axis=1))
    else:
        # data came in as an hanoi_rapid_thumb_midi_3 dimension array (20000x5x2080),
        return np.abs(hilbert(data, axis=2))

'''
Functions for Normalizing labels
'''
def normalizeLabels(label, max_val=99.0):
    return label / max_val

'''
Functions for Normalizing angles
'''
def normalizeAngles(angles, max_val=180.0):
    return angles / max_val