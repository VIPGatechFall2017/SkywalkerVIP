'''
`visualizer_util.py`
Lamtharn Hanoi Hantrakul - created 1-23-2018

In order to "animate" the matplotlib graph in real time, we need to create a set of special
functions that continuously updates the graphing window with new data.

Along with `data_visualizer.py` this is by far the messiest and hackiest code in this entire project. I am sorry.
Matplotlib is not fun to play with sometimes.
'''

import argparse
from matplotlib import pyplot as plt, gridspec
import numpy as np
import sys
# for changing tick size
import matplotlib as mpl
label_size = 6
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size


'''
###########################################################
#################### FOR DATA PLAYBACK ####################
###########################################################
'''
def init_fig(data, proc_data, labels, raw_angles, fontsize, mode):
    '''
    We first initialize the figures with fixed dimensions for the graphs
    :param data: (5x2080) numpy array ~ raw data
    :param proc_data: (5x2080) numpy array ~ processed data
    :param labels: (1x5) numpy array
    :param raw_angles: (1x3) numpy array
    :param mode: 'full' or 'mini' ~ in "full" all signals are shown | in "mini" only the condensed pseudoimage is shown
    :return: plot_obj (tuple) ~ contains all the hacked matplotlib objects for plotting
    '''
    if mode == 'full':
        # data is (5,540) since we are displaying frame by frame. No need to make (1,5,540) since we are not predicting from a model
        w, h = np.shape(data)
        _, h_proc = np.shape(proc_data)

        fig = plt.figure()
        gs = gridspec.GridSpec(6,4)

        #gs.update(wspace=0.4)

        axs = []
        lines = []
        axs_proc = []
        lines_proc = []

        ticks_list = [0, h / 4, h / 2, 3 * h / 4, h]
        tick_interval = 25

        # for raw data
        for i in range(w):
            ax = fig.add_subplot(gs[i,0:2])  # remember that gridspec is indexed starting from 0
            l, = ax.plot(data[i,:])  # this returns a tuple!

            ax.set_ylim(np.min(data),np.max(data))
            ax.set_xlim(0, h)

            ax.xaxis.set_ticks(ticks_list)
            ax.set_xticklabels([('%i \n (%.i us, %.1f cm)' % (x, x * tick_interval/1000, x * (1.5/2) * tick_interval/10000)) for x in ticks_list])

            lines.append(l)
            axs.append(ax)

        # for processed data
        for i in range(w):
            ax = fig.add_subplot(gs[i,2:])
            l, = ax.plot(proc_data[i,:])  # this returns a tuple!

            ax.set_ylim(np.min(proc_data),np.max(proc_data))
            ax.set_xlim(0, h_proc)

            lines_proc.append(l)
            axs_proc.append(ax)

        # these titles are the same in all frames
        axs[0].set_ylabel("Elem 1", fontdict={'fontsize': fontsize})
        axs[1].set_ylabel("Elem 2", fontdict={'fontsize': fontsize})
        axs[2].set_ylabel("Elem 3", fontdict={'fontsize': fontsize})
        axs[3].set_ylabel("Elem 4", fontdict={'fontsize': fontsize})
        axs[4].set_ylabel("Elem 5", fontdict={'fontsize': fontsize})

        axs_proc[0].set_ylabel("Elem 1", fontdict={'fontsize': fontsize})
        axs_proc[1].set_ylabel("Elem 2", fontdict={'fontsize': fontsize})
        axs_proc[2].set_ylabel("Elem 3", fontdict={'fontsize': fontsize})
        axs_proc[3].set_ylabel("Elem 4", fontdict={'fontsize': fontsize})
        axs_proc[4].set_ylabel("Elem 5", fontdict={'fontsize': fontsize})

        # for raw angles
        ind_angles = ('X','Y','Z')
        y_pos = np.arange(len(ind_angles))
        ax_angles = fig.add_subplot(gs[5,0])  # gs indexing starts from 0!
        (x_ang,y_ang,z_ang) = ax_angles.barh(y_pos,raw_angles,height=0.5,align='center', color='green')
        angles_plot = (x_ang,y_ang,z_ang)
        ax_angles.set_xlim(0,180)
        ax_angles.set_yticks(y_pos)
        ax_angles.set_yticklabels(ind_angles)
        ax_angles.invert_yaxis()

        # for raw labels
        ind = np.arange(1,6)
        ax_labels = fig.add_subplot(gs[5,1])  # gs indexing starts from 0!
        (f0,f1,f2,f3,f4) = ax_labels.bar(ind,labels,color='red')
        labels_plot = (f0,f1,f2,f3,f4)
        ax_labels.set_ylim(0,99)
        ax_labels.set_xticks(ind)

        # for the grayscale image
        ax_grayscale = fig.add_subplot(gs[5, 2:])  # gs indexing starts from 0!
        ax_grayscale.imshow(proc_data, aspect='auto')  # imshow does not return an iterable tuple. Without `aspect` argument the plot gets completely squished


        # init the figure
        plt.tight_layout()
        fig.show()
        plot_obj = (fig, axs, lines, axs_proc, lines_proc, ax_labels, labels_plot, ax_angles, angles_plot, ax_grayscale)
        return plot_obj

    elif mode == 'mini':
        # data is (5,540) since we are displaying frame by frame. No need to make (1,5,540) since we are not predicting from a model
        _, h_proc = np.shape(proc_data)

        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2)

        gs.update(wspace=0.4)

        # for raw angles
        ind_angles = ('X', 'Y', 'Z')
        y_pos = np.arange(len(ind_angles))
        ax_angles = fig.add_subplot(gs[0, 0])  # gs indexing starts from 0!
        (x_ang, y_ang, z_ang) = ax_angles.barh(y_pos, raw_angles, height=0.5, align='center', color='green')
        angles_plot = (x_ang, y_ang, z_ang)
        ax_angles.set_xlim(0, 180)
        ax_angles.set_yticks(y_pos)
        ax_angles.set_yticklabels(ind_angles)
        ax_angles.invert_yaxis()

        # for raw labels
        ind = np.arange(1, 6)
        ax_labels = fig.add_subplot(gs[0, 1])  # gs indexing starts from 0!
        (f0, f1, f2, f3, f4) = ax_labels.bar(ind, labels, color='red')
        labels_plot = (f0, f1, f2, f3, f4)
        ax_labels.set_ylim(0, 99)
        ax_labels.set_xticks(ind)

        # for the grayscale image
        ax_grayscale = fig.add_subplot(gs[1, :])  # gs indexing starts from 0!
        ax_grayscale.imshow(proc_data, aspect='auto')  # imshow does not return an iterable tuple. Without `aspect` argument the plot gets completely squished

        # init the figure
        # plt.tight_layout()
        fig.show()
        plot_obj = (fig, ax_labels, labels_plot, ax_angles, angles_plot, ax_grayscale)
        return plot_obj

def update_fig(plot_obj, frame, data, proc_data, label, angle, mode):
    '''
    In this function, we can update the figure defined in `init_fig()` with data streaming in per frame
    :param frame: (int) ~ frame number
    :param plot_obj: (tuple) ~ contains all the hacked matplotlib objects for plotting
    :param data: (5x2080) numpy array ~ raw data
    :param proc_data: (5x2080) numpy array ~ processed data
    :param labels: (1x5) numpy array
    :param raw_angles: (1x3) numpy array
    :param mode: 'full' or 'mini' ~ in "full" all signals are shown | in "mini" only the condensed pseudoimage is shown
    :return: 0 if succesful
    '''
    if mode == 'full':
        (fig, axs, lines, axs_proc, lines_proc, ax_labels, labels_plot, ax_angles, angles_plot, ax_grayscale) = plot_obj
        # fontsize for each visualizer component
        fontsize = 8

        # update pysonix data display
        axs[0].set_title('Raw data frame: ' + str(frame), fontdict={'fontsize': fontsize})
        for i, (ax, l) in enumerate(zip(axs, lines)):
            l.set_ydata(data[i, :])

        # update processed data display
        axs_proc[0].set_title("Preprocessing Pipeline Result", fontdict={'fontsize': fontsize})
        for i, (ax, l) in enumerate(zip(axs_proc, lines_proc)):
            l.set_ydata(proc_data[i, :])

        # update angles data display
        ax_angles.set_title('Angles', fontdict={'fontsize': fontsize})
        (x_ang, y_ang, z_ang) = angles_plot
        x_ang.set_width(angle[0])
        y_ang.set_width(angle[1])
        z_ang.set_width(angle[2])

        # update labels data display
        ax_labels.set_title('Ground Truth', fontdict={'fontsize': fontsize})
        (f0, f1, f2, f3, f4) = labels_plot
        f0.set_height(label[0])
        f1.set_height(label[1])
        f2.set_height(label[2])
        f3.set_height(label[3])
        f4.set_height(label[4])

        # update the greyscale image
        ax_grayscale.set_title('greyscale "pseudo-image" ', fontdict={'fontsize': fontsize})
        ax_grayscale.imshow(proc_data, aspect='auto')

        fig.canvas.draw()
        fig.canvas.flush_events()
        return 0

    elif mode == 'mini':
        (fig, ax_labels, labels_plot, ax_angles, angles_plot, ax_grayscale) = plot_obj
        # fontsize for each visualizer component
        fontsize = 8

        # update angles data display
        ax_angles.set_title('Angles', fontdict={'fontsize': fontsize})
        (x_ang, y_ang, z_ang) = angles_plot
        x_ang.set_width(angle[0])
        y_ang.set_width(angle[1])
        z_ang.set_width(angle[2])

        # update labels data display
        ax_labels.set_title('Ground Truth', fontdict={'fontsize': fontsize})
        (f0, f1, f2, f3, f4) = labels_plot
        f0.set_height(label[0])
        f1.set_height(label[1])
        f2.set_height(label[2])
        f3.set_height(label[3])
        f4.set_height(label[4])

        # update the greyscale pseudo-image
        ax_grayscale.set_title('greyscale "pseudo-image" frame: ' + str(frame), fontdict={'fontsize': fontsize})
        ax_grayscale.imshow(proc_data, aspect='auto')

        fig.canvas.draw()
        fig.canvas.flush_events()
        return 0

'''
###########################################################
################### FOR MODEL PLAYBACK ####################
###########################################################
'''
def init_model_fig(data, proc_data, labels, raw_angles, predictions, fontsize, mode):

    if mode == 'full':
        w, h = np.shape(data)
        _, h_proc = np.shape(proc_data)

        fig = plt.figure()
        gs = gridspec.GridSpec(6,4) # (5,2)

        #gs.update(wspace=0.25)

        axs = []
        lines = []
        axs_proc = []
        lines_proc = []

        ticks_list = [0, h / 4, h / 2, 3 * h / 4, h]
        tick_interval = 25

        # for raw data
        for i in range(w):
            ax = fig.add_subplot(gs[i,0:2])
            l, = ax.plot(data[i,:])

            ax.set_ylim(np.min(data),np.max(data))
            ax.set_xlim(0, h)

            ax.xaxis.set_ticks(ticks_list)
            ax.set_xticklabels([('%i \n (%.i us, %.1f cm)' % (x, x * tick_interval/1000, x * (1.5/2) * tick_interval/10000)) for x in ticks_list])

            lines.append(l)
            axs.append(ax)

        # for processed data
        for i in range(w):
            ax = fig.add_subplot(gs[i,2:])
            l, = ax.plot(proc_data[i,:])

            ax.set_ylim(np.min(proc_data),np.max(proc_data))
            ax.set_xlim(0, h_proc)

            lines_proc.append(l)
            axs_proc.append(ax)

        # these titles are the same in all frames
        axs[0].set_ylabel("Elem 1", fontdict={'fontsize': fontsize})
        axs[1].set_ylabel("Elem 2", fontdict={'fontsize': fontsize})
        axs[2].set_ylabel("Elem 3", fontdict={'fontsize': fontsize})
        axs[3].set_ylabel("Elem 4", fontdict={'fontsize': fontsize})
        axs[4].set_ylabel("Elem 5", fontdict={'fontsize': fontsize})

        axs_proc[0].set_ylabel("Elem 1", fontdict={'fontsize': fontsize})
        axs_proc[1].set_ylabel("Elem 2", fontdict={'fontsize': fontsize})
        axs_proc[2].set_ylabel("Elem 3", fontdict={'fontsize': fontsize})
        axs_proc[3].set_ylabel("Elem 4", fontdict={'fontsize': fontsize})
        axs_proc[4].set_ylabel("Elem 5", fontdict={'fontsize': fontsize})

        # for raw angles
        ind_angles = ('X','Y','Z')
        y_pos = np.arange(len(ind_angles))
        ax_angles = fig.add_subplot(gs[5,0])  # gs indexing starts from 0!
        (x_ang,y_ang,z_ang) = ax_angles.barh(y_pos,raw_angles,height=0.5,align='center', color='green')
        angles_plot = (x_ang,y_ang,z_ang)
        ax_angles.set_xlim(0,180)
        ax_angles.set_yticks(y_pos)
        ax_angles.set_yticklabels(ind_angles)
        ax_angles.invert_yaxis()

        # for raw labels
        ind = np.arange(1,6)
        ax_labels = fig.add_subplot(gs[5,1])  # gs indexing starts from 0!
        (f0,f1,f2,f3,f4) = ax_labels.bar(ind,labels,color='red')
        labels_plot = (f0,f1,f2,f3,f4)
        ax_labels.set_ylim(0,99)
        ax_labels.set_xticks(ind)

        # for predicted labels
        ax_pred = fig.add_subplot(gs[5,3])  # gs indexing starts from 0!
        (p0,p1,p2,p3,p4) = ax_pred.bar(ind,predictions,color='red')
        pred_plot = (p0,p1,p2,p3,p4)
        ax_pred.set_ylim(0,99)
        ax_pred.set_xticks(ind)

        # init the figure
        plt.tight_layout()
        fig.show()

        return_tuple = (fig, axs, lines, axs_proc, lines_proc, ax_labels, labels_plot, angles_plot, ax_angles, ax_pred, pred_plot)
        return return_tuple

    elif mode == 'mini':
        _, h_proc = np.shape(proc_data)

        fig = plt.figure()
        gs = gridspec.GridSpec(1, 3)  # (5,2)

        gs.update(wspace=0.25)

        axs = []
        lines = []
        axs_proc = []
        lines_proc = []

        # for raw angles
        ind_angles = ('X', 'Y', 'Z')
        y_pos = np.arange(len(ind_angles))
        ax_angles = fig.add_subplot(gs[0, 0])  # gs indexing starts from 0!
        (x_ang, y_ang, z_ang) = ax_angles.barh(y_pos, raw_angles, height=0.5, align='center', color='green')
        angles_plot = (x_ang, y_ang, z_ang)
        ax_angles.set_xlim(0, 180)
        ax_angles.set_yticks(y_pos)
        ax_angles.set_yticklabels(ind_angles)
        ax_angles.invert_yaxis()

        # for raw labels
        ind = np.arange(1, 6)
        ax_labels = fig.add_subplot(gs[0, 1])  # gs indexing starts from 0!
        (f0, f1, f2, f3, f4) = ax_labels.bar(ind, labels, color='red')
        labels_plot = (f0, f1, f2, f3, f4)
        ax_labels.set_ylim(0, 99)
        ax_labels.set_xticks(ind)

        # for predicted labels
        ax_pred = fig.add_subplot(gs[0, 2])  # gs indexing starts from 0!
        (p0, p1, p2, p3, p4) = ax_pred.bar(ind, predictions, color='red')
        pred_plot = (p0, p1, p2, p3, p4)
        ax_pred.set_ylim(0, 99)
        ax_pred.set_xticks(ind)

        # init the figure
        # plt.tight_layout()
        fig.show()

        return_tuple = (fig, ax_labels, labels_plot, angles_plot, ax_angles, ax_pred, pred_plot)
        return return_tuple


def update_model_fig(plot_obj, frame, data, proc_data, label, angle, prediction, mode):
    if mode == 'full':
        (fig, axs, lines, axs_proc, lines_proc, ax_labels, labels_plot, angles_plot, ax_angles, ax_pred, pred_plot) = plot_obj

        # fontsize for each visualizer component
        fontsize = 8

        # update pysonix data display
        axs[0].set_title('Raw data frame: ' + str(frame), fontdict={'fontsize': fontsize})
        for i, (ax, l) in enumerate(zip(axs, lines)):
            l.set_ydata(data[i, :])

        # update processed data display
        axs_proc[0].set_title("Model view", fontdict={'fontsize': fontsize})
        for i, (ax, l) in enumerate(zip(axs_proc, lines_proc)):
            l.set_ydata(proc_data[i, :])

        # update angles data display
        ax_angles.set_title('Angles', fontdict={'fontsize': fontsize})
        (x_ang,y_ang,z_ang) = angles_plot
        x_ang.set_width(angle[0])
        y_ang.set_width(angle[1])
        z_ang.set_width(angle[2])

        # update labels data display
        ax_labels.set_title('Ground Truth', fontdict={'fontsize': fontsize})
        (f0,f1,f2,f3,f4) = labels_plot
        f0.set_height(label[0])
        f1.set_height(label[1])
        f2.set_height(label[2])
        f3.set_height(label[3])
        f4.set_height(label[4])

        # update predicted data display
        ax_pred.set_title('Prediction', fontdict={'fontsize': fontsize})
        (p0,p1,p2,p3,p4) = pred_plot
        p0.set_height(prediction[0])
        p1.set_height(prediction[1])
        p2.set_height(prediction[2])
        p3.set_height(prediction[3])
        p4.set_height(prediction[4])

        fig.canvas.draw()
        fig.canvas.flush_events()

    elif mode == "mini":
        fig, ax_labels, labels_plot, angles_plot, ax_angles, ax_pred, pred_plot = plot_obj

        # fontsize for each visualizer component
        fontsize = 8

        # update angles data display
        ax_angles.set_title('Angles', fontdict={'fontsize': fontsize})
        (x_ang, y_ang, z_ang) = angles_plot
        x_ang.set_width(angle[0])
        y_ang.set_width(angle[1])
        z_ang.set_width(angle[2])

        # update labels data display
        ax_labels.set_title('Ground Truth Frame: ' + str(frame), fontdict={'fontsize': fontsize})
        (f0, f1, f2, f3, f4) = labels_plot
        f0.set_height(label[0])
        f1.set_height(label[1])
        f2.set_height(label[2])
        f3.set_height(label[3])
        f4.set_height(label[4])

        # update predicted data display
        ax_pred.set_title('Prediction', fontdict={'fontsize': fontsize})
        (p0, p1, p2, p3, p4) = pred_plot
        p0.set_height(prediction[0])
        p1.set_height(prediction[1])
        p2.set_height(prediction[2])
        p3.set_height(prediction[3])
        p4.set_height(prediction[4])

        fig.canvas.draw()
        fig.canvas.flush_events()

'''
###########################################################
############# FOR MODEL TIMEGRAPH PLAYBACK   #############
###########################################################
'''

def init_dataplayer_fig(label_data, pred_data, angle_data, fontsize):

    w, h = np.shape(label_data)

    fig = plt.figure()
    gs = gridspec.GridSpec(6,1)

    #gs.update(wspace=2.0)

    axs = []
    lines = []
    lines_pred = []

    # for each finger
    for i in range(w):
        ax = fig.add_subplot(gs[i,:])
        l, = ax.plot(label_data[i,:], label="truth") # plot ground truth data in blue
        l2, = ax.plot(pred_data[i,:], label="pred", color='red') # we want to plot predicted data superimposed over ground truth data
        ax.legend(loc=6, fontsize=6)

        ax.set_ylim(-5, 105)
        ax.set_xlim(0, h)
        ticks_list = [0, h / 4, h / 2, 3 * h / 4, h]
        pulse_rate = 70
        ax.xaxis.set_ticks(ticks_list)
        ax.set_xticklabels([('%.i sec'%(x/pulse_rate)) for x in ticks_list])

        lines.append(l)
        lines_pred.append(l2)
        axs.append(ax)

    # these titles are the same in all frames
    axs[0].set_ylabel("THUMB", fontdict={'fontsize': fontsize})
    axs[1].set_ylabel("INDEX", fontdict={'fontsize': fontsize})
    axs[2].set_ylabel("MIDDLE", fontdict={'fontsize': fontsize})
    axs[3].set_ylabel("RING", fontdict={'fontsize': fontsize})
    axs[4].set_ylabel("PINKY", fontdict={'fontsize': fontsize})

    # for plotting angles
    ax_angle = fig.add_subplot(gs[5,:])
    ax_angle.set_ylim(0, 180)
    ax_angle.set_xlim(0, h)
    ticks_list = [0, h / 4, h / 2, 3 * h / 4, h]
    pulse_rate = 70
    ax_angle.xaxis.set_ticks(ticks_list)
    ax_angle.set_xticklabels([('%.i sec' % (x / pulse_rate)) for x in ticks_list])
    ax_angle.set_ylabel("(Angle)", fontdict={'fontsize': fontsize})

    lx, = ax_angle.plot(angle_data[0], label="x") # you need the comma because this function returns a tuple!!!!
    ly, = ax_angle.plot(angle_data[1], label="y")# you need the comma because this function returns a tuple!!!!
    lz, = ax_angle.plot(angle_data[2], label="z") # you need the comma because this function returns a tuple!!!!
    ax_angle.legend(loc=6, fontsize=6)
    lines_angle = (lx,ly,lz)

    fig.show()

    fig.tight_layout()

    return_tuple = (fig, axs, lines, lines_pred, lines_angle)
    return return_tuple