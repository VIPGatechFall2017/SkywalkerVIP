# Matthew Kaufer

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

import csv

data = np.array(list(csv.reader(open('arrhythmia.data.csv'))))

first_size = len(data[0])

for i in range(len(data)):
    if len(data[i]) != first_size:
        print("Uh oh, jagged array", len(data[i]), first_size)
        return
    for j in range(len(data[i])):
        # have do to some sketchy data cleanup here - for variables the dataset doesn't know,
        # it replcaes the number with a ?, so I assume 0 here, which may or may not be correct
        if data[i][j] == '?':
            data[i][j] = 0
        else:
            data[i][j] = float(data[i][j])

X = data[:,:-1]
# everything but last column to be our X matrix
y = data[:,-1]
# last column equal to our Y matrix

def train_svc(kernel='rbf', C=10, gamma=5):
    return svm.SVC(kernel=kernel, C=C,gamma=gamma).fit(X, y)

def svc_accuracy(svc):
    fails = 0

    for i in range(len(X)):
        if svc.predict([X[i]])[0] != y[i]:
            fails+=1
    return fails


svc = train_svc()
print("Total misclassifications", svc_accuracy(svc))

# I should split the data into test data and training data
# but there are a large amount of Y possibilities with very few
# entries for each value
# So at the moment, we're just classifying