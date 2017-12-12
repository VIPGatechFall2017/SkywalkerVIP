import numpy as np
import pandas as pd
import glob, re, datetime, os, pickle, cv2, itertools
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

def get_distribution(labels, plot=False):
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts)) # record the label distribution
    if plot:
        objects = dist.keys()
        y_pos = np.arange(len(objects))
        x = dist.values()
        plt.bar(y_pos, x, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.title('Distribution')
        plt.show()
    return dist

def get_labels(path):
    df = pd.read_csv(path, sep=';', header=None, delimiter=' ')
    df = df.drop(0, axis=1)
    df[5] = df[5].map(lambda x: x.rstrip(';'))
    df[5] = df[5].map(lambda x: int(x))
    data = np.array(df)
    data = (data>=30).astype(int) 
    conM = np.array([[2**4], [2**3], [2**2], [2**1], [2**0]])
    labels = np.dot(data, conM)
    return labels

def get_data(labels_old, path):
    loc = path
    addrs = glob.glob(loc)
    regex = re.compile(r'\d+')
    labels, imgs = [], []
    t0 = datetime.datetime.now()
    if os.path.isfile("imgs.pkl") \
    and os.path.isfile("labels.pkl"):
        imgs = pickle.load(open("imgs.pkl", 'rb'))
        labels = pickle.load(open("labels.pkl", 'rb'))
    else:
        for img in addrs:
            index = int(regex.findall(img)[1])
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC) #resize
            imgs.append(image)
            labels.append(labels_old[index][0])
        pickle.dump(imgs, open("imgs.pkl", 'wb'))
        pickle.dump(labels, open("labels.pkl", 'wb'))
    print ("Reading time:", datetime.datetime.now()-t0)
    return np.array(imgs), np.array(labels)

def subsampling(data, labels, threshold=-1, seed=0):
    data, labels = shuffle(data, labels, random_state=seed)
    dist = get_distribution(labels)
    _data, _labels, count = [], [], dict()
    if threshold == -1:
        threshold = np.median([v for v in dist.values()])
    for k in dist.keys():
        count[k] = 0
    for i in range(0, len(labels)):
        if count[labels[i]] < threshold:
            _data.append(data[i])
            _labels.append(labels[i])
            count[labels[i]] += 1
    _data, _labels = shuffle(_data, _labels, random_state=seed)
    return np.array(_data), np.array(_labels)

def plot_images(images, cls_true, cls_pred=None):
    random_indices = random.sample(range(len(images)), min(len(images), 4))
    fig, axes = plt.subplots(2, 2)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def plot_conv_weights(weights, input_channel=0):
    w = sess.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)
    num_filters = w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def plot_conv_layer(layer, image):
    image = image.reshape(1, img_size_flat)
    feed_dict = {x: image}
    values = sess.run(layer, feed_dict=feed_dict)
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()