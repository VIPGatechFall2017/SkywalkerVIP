# system lib
import datetime
import os

# ML lib
from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

# plot lib
import matplotlib.pyplot as plt

# image processing lib
import cv2

# util lib
import numpy as np
import pandas as pd
import itertools
import re
import glob
import pickle
from collections import Counter

def read_data():
    df = pd.read_csv('/Users/Raymond/download/zach_cont_gest_1/zach_cont_gest_1/data.txt', sep=';', header=None, delimiter=' ')
    df = df.drop(0, axis=1)
    df[5] = df[5].map(lambda x: x.rstrip(';'))
    df[5] = df[5].map(lambda x: int(x))
    data = np.array(df)
    data = (data>=30).astype(int) # threshold 30, if >30 => 1, otherwise, 0.
    # convert a binary matrix (n*5) to a decimal matrix (n*1) ie. 11111 => 31, 11110 => 30
    act = np.array([[2**4], [2**3], [2**2], [2**1], [2**0]])
    ret = np.dot(data, act)

    unique, counts = np.unique(ret, return_counts=True)
    dist = dict(zip(unique, counts)) # record the label distribution
    return ret, dist

def plot_distribution(labels, dist):
    objects = dist.keys()
    y_pos = np.arange(len(objects))
    x = dist.values()
    plt.bar(y_pos, x, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.title('Distribution')
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def spliting(pngs, labels, dist, per=0.7, random=0, balance=True):
    t0 = datetime.datetime.now()
    mean = np.mean(dist.values())
    count = dict()
    new_labels, new_pngs = [], []
    # re-balance the dataset based on the mean
    if balance:
        for k in dist.keys():
            count[k] = 0
        for i in range(0, len(labels)):
            if count[labels[i]] < mean:
                new_pngs.append(pngs[i])
                new_labels.append(labels[i])
                count[labels[i]] += 1
    else:
        new_labels, new_pngs = labels, pngs
    new_labels, new_pngs = shuffle(np.array(new_labels), np.array(new_pngs), random_state=1)

    unique, counts = np.unique(new_labels, return_counts=True)
    new_dist = dict(zip(unique, counts))

    split = int(len(new_labels)*per)
    sp = datetime.datetime.now()-t0
    print "Spliting time:", sp
    return new_pngs[:split, :], new_pngs[split:, :], new_labels[:split], new_labels[split:]

def pred_svm(x_train, x_test, y_train, y_test, kernel='poly', degree=3, gamma='auto', verbose=1):
    # Support vector classification.
    t0 = datetime.datetime.now()
    classifier = svm.SVC(kernel=kernel, degree=degree, gamma=gamma, probability=True, random_state=0) #gamma=0.001, degree=3, kernel='poly','rbf','linear'
    classifier.fit(x_train, y_train)
    trT = datetime.datetime.now()-t0
    t0 = datetime.datetime.now()
    expected = y_test
    predicted = classifier.predict(x_test)
    scoreI = classifier.score(x_train, y_train)
    scoreO = classifier.score(x_test, y_test)
    teT = datetime.datetime.now()-t0
    if verbose == 1:
        print "####SVM model####"
        print "Training time:", trT
        print "Testing time:", teT
        print("\nTraining Accuracy: {:.2%}".format(scoreI))
        print("Testing Accuracy: {:.2%}\n".format(scoreO))
        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected, predicted)))
        cnf_matrix = confusion_matrix(expected, predicted)
        np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        plt.figure()
        unique, counts = np.unique(y_train, return_counts=True)
        plot_confusion_matrix(cnf_matrix, classes=unique,
                              title='Confusion matrix, without normalization')
        plt.show()
    return scoreI, scoreO, classifier

def testing(label_set, dist):
    loc = '/Users/Raymond/download/zach_cont_gest_1/zach_cont_gest_1/*.png'
    addrs = glob.glob(loc)
    regex = re.compile(r'\d+')
    labels, pngs = [], []
    t0 = datetime.datetime.now()
    if os.path.isfile("pngs.pkl") \
    and os.path.isfile("labels.pkl") \
    and os.path.isfile("dist.pkl"):
        pngs = pickle.load(open("pngs.pkl", 'r'))
        labels = pickle.load(open("labels.pkl", 'r'))
        dist = pickle.load(open("dist.pkl", 'r'))
    else:
        for img in addrs:
            index = int(regex.findall(img)[2]) # index depends on how many int inside the path
            labels.append(label_set[index][0])
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC) #resize
            pngs.append(image)
        pickle.dump(pngs, open("pngs.pkl", 'wb'))
        pickle.dump(labels, open("labels.pkl", 'wb'))
        pickle.dump(dist, open("dist.pkl", 'wb'))
    print "Reading time:", datetime.datetime.now()-t0
    # convert list to nparray
    labels = np.array(labels)
    pngs = np.array(pngs)
    length = pngs.shape[0]
    # reshape it from 2D to 1D
    pngs = pngs.reshape((length, -1))
    x_train, x_test, y_train, y_test = spliting(pngs, labels, dist, balance=True)
    # pred_svm(x_train, x_test, y_train, y_test)
    full_analysis(x_train, x_test, y_train, y_test, kernel='rbf')

def full_analysis(x_train, x_test, y_train, y_test, kernel='poly'):
    if kernel == 'poly':
        svm_train_dict = dict()
        svm_test_dict = dict()
        for d in range(1, 10):
            trP, teP, _ = pred_svm(x_train, x_test, y_train, y_test, kernel='poly', degree=d, verbose=0)
            svm_train_dict[d] = trP
            svm_test_dict[d] = teP
            
        plt.clf()   # clear figure
        plt.plot(svm_train_dict.keys(), svm_train_dict.values(), '-', label='Training acc')
        plt.plot(svm_test_dict.keys(), svm_test_dict.values(), '--', label='Testing acc')
        plt.title('(SVM) Training and testing accuracy')
        plt.xlabel('Degree')
        plt.ylabel('Acc')
        plt.legend()
        plt.show()
    elif kernel == 'rbf':
        svm_train_dict = dict()
        svm_test_dict = dict()
        g = np.linspace(0.1, 0.00001, num=10)
        for idx in range(1, 10):
            trP, teP, _ = pred_svm(x_train, x_test, y_train, y_test, kernel='rbf', gamma=g[idx], verbose=0)
            svm_train_dict[idx] = trP
            svm_test_dict[idx] = teP
        x = map(str, svm_train_dict.keys())     
        plt.clf()   # clear figure
        plt.plot(x, svm_train_dict.values(), '-', label='Training acc')
        plt.plot(x, svm_test_dict.values(), '--', label='Testing acc')
        plt.title('(SVM) Training and testing accuracy')
        plt.xlabel('Gamma')
        plt.ylabel('Acc')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    label_set, dist = read_data()
    # import pdb;pdb.set_trace()
    testing(label_set, dist)