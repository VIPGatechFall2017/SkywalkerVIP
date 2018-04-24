''' Modified based on Hanoi's code'''

import matplotlib
matplotlib.use('TkAgg')

from config_visualizer import *
from visualizer_util import *
import glob, re
import numpy as np
from utils import *

plt.ion()

fingers = ['thumb']#,'mixed']#['thumb', 'pinky', 'mixed', 'index', 'middle', 'ring']
types = ['train']#, 'test']
ids = [1]
regex = re.compile(r'\d+')
train_set = dict()
test_set = dict()
for finger in fingers:
    for t in types:
        for name in glob.glob('data/*%s*%s'%(t, finger)):
            curr_id = int(regex.findall(name)[3])
            if curr_id in ids:
                dw, da, dl = unpackSingle_h5(name)
                if t == 'train':
                    if len(train_set.keys()) == 0:
                        train_set['dw'] = dw
                        train_set['da'] = da
                        train_set['dl'] = dl
                    else:
                        train_set['dw'] = np.concatenate((train_set['dw'], dw))
                        train_set['da'] = np.concatenate((train_set['da'], da))
                        train_set['dl'] = np.concatenate((train_set['da'], da))
                else:
                    if len(test_set.keys()) == 0:
                        test_set['dw'] = dw
                        test_set['da'] = da
                        test_set['dl'] = dl
                    else:
                        test_set['dw'] = np.concatenate((test_set['dw'], dw))
                        test_set['da'] = np.concatenate((test_set['da'], da))
                        test_set['dl'] = np.concatenate((test_set['da'], da))


offset = 0
mode = 'mini'
fontsize = 8
plot_obj = None # Stores a tuple of axes and figure objects
print("Playing with offset", offset, "in mode:", mode)
num_frames = dw.shape[0]
for i in range(len(train_set['dw'])):
    data = pysonixVisualizer(dw[i])  # raw data as seen on pysonix
    proc_data = processData(dw[i])  # processed data (what the model will see)
    label = dl[i] # raw labels
    angle = da[i] # raw angles

    if (i == 0):
        # if this is the very first frame, then initialize the figure with fixed width and heights
        plot_obj = init_fig(data,proc_data,label,angle, fontsize, mode)
    else:
        # simply update the defined figure with new data
        update_fig(plot_obj, offset+i, data, proc_data, label, angle, mode)

    #time.sleep(1e-8) # this didn't really change anything...