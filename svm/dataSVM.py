# system lib
import datetime
import os
import sys

# ML lib
from sklearn import svm, metrics
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from scipy.misc import imsave

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

##############################################################
##  mcw0805's edited version for SVM group in RMVIPFall2017 ##
##############################################################
# original credit to raymondhehe

"""

######  #     #    #     # ### ######  
#     # ##   ##    #     #  #  #     # 
#     # # # # #    #     #  #  #     # 
######  #  #  #    #     #  #  ######  
#   #   #     #     #   #   #  #       
#    #  #     #      # #    #  #       
#     # #     #       #    ### #       


    |~~~~~~~~~~~~~|
    |~~~~~~~~~~~~~|
    |             |
/~~\|         /~~\|
\__/          \__/         


     ; 
     ;;
     ;';.
     ;  ;;
     ;   ;;
     ;    ;;
     ;    ;;
     ;   ;'
     ;  ' 
,;;;,; 
;;;;;;
`;;;;'

"""

# videoPath: where hand.mov is located on your machine
# format is /../vidFileNameHere.mov
#
# imgPath: where the converted img files will be stored 
# format is /../.. (no extensions at the end)
#
# will convert .mov to frames of .png (other img files)
def mov_to_png(videoPath, imgPath):
    cap = cv2.VideoCapture(videoPath)

    # extract video file name from tail
    base, fileName = os.path.split(videoPath)

    frame_num = 0

    if not os.path.isdir(imgPath):
        print 'New directory ' + imgPath + ' has been created'
        cmd = 'mkdir ' + imgPath
        os.system(cmd)

    # start time
    t0 = datetime.datetime.now()
    print 'Video reading start time: ', t0

    while(1):
        ret, frame = cap.read()

        if ret:
            save_path = imgPath + "/" + fileName + '_'+str(frame_num) + '.png'

            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pil_img = Image.fromarray(gray_img)

            box = (75, 0, 535, 460)
            pil_img = pil_img.crop(box)

            pil_img = pil_img.resize((28, 28))
            pil_img.save(save_path)

            frame_num += 1

        else:
            break;

    # calculate total reading time: endTime - initialTime
    end = datetime.datetime.now()-t0
    print 'Video convert total time: ', end


# textPath: where data.txt is located on your machine
# threshold: must be [0, 99]; number greater or equal to the specified
# bound will be classified as bent
#
# data.txt format is as follows (in the quotes,
# ignore spaces before/after the quote):
# " #, x x x x x; "
# # (image number) starts at 0 and it increases by one each row
# 0 <= x <= 99
# leftmost val of x corresponds to the thumb
# rightmost val of x corresponds to the pinky
#
# reads the TEXT FILE. --> format is /../fileNameHere.txt
def read_data(textPath, threshold=30):

    # start time
    t0 = datetime.datetime.now()
    print 'Reading start time: ', t0

    # read in the file
    df = pd.read_csv(textPath, sep=';', header=None, delimiter=' ')

    # gets rid of the row number
    df = df.drop(0, axis=1)

    # gets rid of the semicolon at the end and converts the string into int
    df[5] = df[5].map(lambda x: x.rstrip(';'))
    df[5] = df[5].map(lambda x: int(x))

    # create a 2d array
    # [[#, #, #, #, #] 
    # [#, #, #, #, #]
    # etc ]
    # size = imgCount x 5
    data = np.array(df)
    print 'data shape', np.shape(data)

    # data produces logical array (0s and 1s) based on the threshold
    data = (data>=threshold).astype(int)
    #print 'data: ', data
    #print 'length of data: ', len(data)

    # base 2 power col vector
    # size = 5 x 1
    act = np.array([[2**4], [2**3], [2**2], [2**1], [2**0]])

    # row x 1 matrix, where leftmost val turns into 2^4
    # row = number of images
    # format: [[], [], [], ...]
    # size of label_set = imgCount x 1
    label_set = np.dot(data, act)

    unique, counts = np.unique(label_set, return_counts=True)

    # finger map
    # maps the finger and the data count for each finger
    # format: {class0 : count0, class1 : count1, class2 : count2, etc.}
    fmap = dict(zip(unique, counts))

    # calculate total reading time: endTime - initialTime
    end = datetime.datetime.now()-t0
    print 'Reading total time: ', end

    return label_set, fmap


# imgPath: pic file format in the directory
# --> format is /../*.png
#
# resultPath: (optional), destination directory of the .pkl files
# function called after the text file is read
#
# default downsize dimension is 28x28 pixels
#
# pickles (basically serializes) the IMAGE FILES
def pickleData(label_set, fmap, imgPath, resultPath="", imgDim=28):
    labels, imgArr = [], []

    if resultPath == "":
        path_to, file_extension = os.path.splitext(imgPath)
        resultPath = path_to
        print resultPath

    imgpkl = os.path.join(resultPath, "imgs.pkl")
    labelspkl = os.path.join(resultPath, "labels.pkl")
    fmappkl = os.path.join(resultPath, "fmap.pkl")

    # has not been pickled yet, go ahead and pickle
    if not os.path.isfile(imgpkl) \
    and not os.path.isfile(labelspkl) \
    and not os.path.isfile(fmappkl):

        t0 = datetime.datetime.now()
        print 'Pickling start time: ', t0

        for img in glob.glob(imgPath):
            originalImg = cv2.imread(img)
            resized = cv2.resize(originalImg, (imgDim, imgDim)) # resize to specified pixel dimension
            gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) # grayscale -> 2D
            imgArr.append(gray_img)

            head, tail = os.path.split(img) # tail is the last thing behind the slash in the path
            index = int(re.findall('\d+', tail )[0]) # grabs number from the img file name
            labels.append(label_set[index][0])

        with open(imgpkl, 'wb') as f:
            pickle.dump(imgArr, f)
            f.close()

        with open(labelspkl, 'wb') as f:
            pickle.dump(labels, f)
            f.close()

        with open(fmappkl, 'wb') as f:
            pickle.dump(fmap, f)
            f.close()

    else:
        print 'Data already pickled.'
        return

    end = datetime.datetime.now() - t0
    print 'Total pickling time: ', end

# plots the distribution of different finger classes
# of the data set being in analysis
def plot_distribution(labels, dist):

    objects = dist.keys()

    y_pos = np.arange(len(objects))
    x = dist.values()

    plt.bar(y_pos, x, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.xlabel('Finger Bend Class')
    plt.ylabel('Counts')
    plt.title('Distribution')
    plt.show()


# splits the images into training and testing sets 
def splitting(pngs, labels, fmap, per=0.7, random=0, balance=True):

    t0 = datetime.datetime.now()
    print 'Start splitting time: ', t0

    mean = np.mean(fmap.values())

    count = dict()
    new_labels, new_pngs = [], []

    # re-balance the dataset based on the mean
    if balance:
        for k in fmap.keys():
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

    split = int(len(new_labels)*per) #70% for training

    sp = datetime.datetime.now()-t0
    print 'Splitting time:', sp

    # order of arrays returned: x_train, x_test, y_train, y_test 
    return new_pngs[:split, :], new_pngs[split:, :], new_labels[:split], new_labels[split:]


# current default model
def build_train_pred(x_train, x_test, y_train, y_test,
                    kernel='poly', degree=3, gamma='auto', verbose=1):


    # record starting training time
    t0 = datetime.datetime.now()
    print 'Training time: ', t0

    classifier = svm.SVC(kernel=kernel, degree=degree, gamma=gamma, probability=True, random_state=0) #gamma=0.001, degree=3, kernel='poly','rbf','linear'
    classifier.fit(x_train, y_train)

    trainTime = datetime.datetime.now()-t0
    print 'Total training time', trainTime

    # record starting testing time
    t0 = datetime.datetime.now()
    print 'Testing time start: ', t0

    expected = y_test
    predicted = classifier.predict(x_test)

    scoreI = classifier.score(x_train, y_train)
    scoreO = classifier.score(x_test, y_test)

    testTime = datetime.datetime.now()-t0
    print 'Total testing time: ', testTime

    # print result
    if verbose == 1:
        print "\n####SVM model####"
        print "Training time:", trainTime
        print "Testing time:", testTime

        print("\nTraining Accuracy: {:.2%}".format(scoreI))
        print("Testing Accuracy: {:.2%}\n".format(scoreO))
        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected, predicted)))

        cnf_matrix = confusion_matrix(expected, predicted)
        np.set_printoptions(precision=2)

        # plot non-normalized confusion matrix
        plt.figure()
        unique, counts = np.unique(y_train, return_counts=True)
        plot_confusion_matrix(cnf_matrix, classes=unique,
                            title='Confusion Matrix, without normalization')
        plt.tight_layout()
        plt.show()

    return scoreI, scoreO, classifier


# plots confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
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


# can feed in different kernel types and different hyperparameters
# in order to build the best model
def analyze(x_train, x_test, y_train, y_test, kernel='poly'):

    t0 = datetime.datetime.now()
    print 'Analyzing... start time: ', t0

    # test 10 different degree values for polynomial
    if kernel == 'poly':
        svm_train_dict = dict()
        svm_test_dict = dict()

        for d in range(1, 10):
            trP, teP, _ = build_train_pred(x_train, x_test, y_train, y_test, kernel='poly', degree=d, verbose=0)
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

    elif kernel == 'rbf': #test different gamma values
        svm_train_dict = dict()
        svm_test_dict = dict()

        # define gamma values
        # evenly space between 0.00001 and 0.1 for 10 values
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

    end = t0-datetime.datetime.now()
    print 'Total analysis time: ', end

# this function is what you want to mess with
# this is the main driver for the model
#
# feed in the data from reading the text file
# pickle thme if not already done
# generate training data set (split data)
# build model, train, and test
def experiment(label_set, fmap, imgPath, resultPath, mode=None, plotDist=False):

    t0 = datetime.datetime.now()
    print 'Experiment start time: ', t0

    # plot the distribution of the data set
    if plotDist:
        plot_distribution(label_set, fmap)

    pickleData(label_set, fmap, imgPath, resultPath)

    imgpkl = os.path.join(resultPath, 'imgs.pkl')
    labelspkl = os.path.join(resultPath, 'labels.pkl')
    fmappkl = os.path.join(resultPath, 'fmap.pkl')

    # load pkl data
    imgs = pickle.load(open(imgpkl, 'r'))
    labels = pickle.load(open(labelspkl, 'r'))
    fmap = pickle.load(open(fmappkl, 'r'))

    # turn them into npy arrays
    labels = np.array(labels)
    imgs = np.array(imgs)

    length = imgs.shape[0]
    imgs = imgs.reshape((length, -1)) # reshape it from 2D to 1D

    # preparing training set
    x_train, x_test, y_train, y_test = splitting(imgs, labels, fmap, balance=True)

    # default mode uses build_train_pred function
    if mode is None:
        build_train_pred(x_train, x_test, y_train, y_test)

    else: # other option is deep analysis of multiple hyperparameters (default to rbf for this experiment)
        analyze(x_train, x_test, y_train, y_test, kernel='rbf')

    end =  datetime.datetime.now()-t0
    print "Total experiment time:", end

if __name__ == "__main__":

    # Basic steps taken:
    #   - read text data
    #   - pickle img data + labels
    #   - split data into training vs. test
    #   - train
    #   - test

    #-----------------------------------------------------#
    # custom local paths: change the next three variables #
    #-----------------------------------------------------#

    # textPath format:  /../fileNameHere.txt (fileNameHere will most likely be data.txt)
    textPath = '/REPLACE/WITH/PATH/HERE/data.txt'

    # textPath format:  /../*.png (all the .png files from the movie)
    imgPath = '/REPLACE/WITH/PATH/HERE/*.png'

    # resultPath format: /../.. (no extensions at the end)
    # this is where you want your .pkl files to be placed at
    resultPath = '/REPLACE/WITH/PATH/HERE'

    # file paths for picked labels and finger map
    labelPklPath = '/REPLACE/WITH/PATH/HERE/labels.pkl'

    fmapPklPath = '/REPLACE/WITH/PATH/HERE/fmap.pkl'

    #--------------------#
    # data science magic #
    #--------------------#
    if not os.path.isfile(labelPklPath) \
    and not os.path.isfile(fmapPklPath):
        print 'Text data processing...'
        label_set, fmap = read_data(textPath)
    else:
        print 'Text data information (labels and finger map) was already pickled.'
        label_set = pickle.load(open(labelPklPath, 'r'))
        fmap = pickle.load(open(fmapPklPath, 'r'))

    experiment(label_set, fmap, imgPath, resultPath)

    #print 'labels', label_set[78:100, :]

