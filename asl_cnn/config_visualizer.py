'''
`config_visualizer.py`
Lamtharn Hanoi Hantrakul - created 1-22-2018
Define global functions for processing and visualizing data. These will be called in `visualize_data.py`
'''

from data_preprocessing_util import *

'''
####################################################
####### CONFIGURATION FOR DATA VISUALIZER ##########
####################################################
'''
#########################################
# Prototype preprocessing function here #
#########################################
def processData(data, verbose=False):
    data = bandpassFilter(data, f_low=2e6, f_high=8e6, verbose=verbose)
    data = envelope(data, verbose=verbose)
    #data = extractCentroid(data, verbose=verbose, plotting=True)
    #data = medianFilter2(data, size=40, verbose=verbose)
    data = normalizePerChannel(data, verbose=verbose)
    return data

###################################################
# Default preprocessing to emulate Pysonix system #
###################################################
def pysonixVisualizer(data, verbose=False):
    data = bandpassFilter(data, f_low=2e6, f_high=8e6, verbose=verbose)
    data = normalizePerChannel(data, verbose=verbose)
    return data