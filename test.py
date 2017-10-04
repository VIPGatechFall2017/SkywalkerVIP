import numpy as np
import pandas as pd
import re
import glob
import cv2
import datetime
import matplotlib.pyplot as plt
import pickle
import os
from collections import Counter
from sklearn import svm, metrics
from sklearn.utils import resample, shuffle

def read_data():
	df = pd.read_csv('./zach_cont_gest_1/data.txt', sep=';', header=None, delimiter=' ')
	df = df.drop(0, axis=1)
	df[5] = df[5].map(lambda x: x.rstrip(';'))
	df[5] = df[5].map(lambda x: int(x))
	data = np.array(df)
	data = (data>=30).astype(int)
	act = np.array([[2**4], [2**3], [2**2], [2**1], [2**0]])
	ret = np.dot(data, act)
	unique, counts = np.unique(ret, return_counts=True)
	dist = dict(zip(unique, counts))
	return ret, dist

def save_img():
	loc = './zach_img/*.png'
	png = []
	count = 0
	for img in glob.glob(loc):
		png.append(io.imread(img))
		count+=1
	np.save('imgs.npy', png)

def load_img():
	return np.load('imgs.npy')

def plot(labels, dist):
	objects = dist.keys()
	y_pos = np.arange(len(objects))
	x = dist.values()
	plt.bar(y_pos, x, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.title('Distribution')
	plt.show()

def spliting(pngs, labels, dist, per=0.7, random=0):
	mean = np.mean(dist.values())
	# mean = np.sum(dist.values())
	count = dict()
	new_labels, new_pngs = [], []
	for k in dist.keys():
		count[k] = 0
	for i in range(0, len(labels)):
		if count[labels[i]] < mean:
			new_pngs.append(pngs[i])
			new_labels.append(labels[i])
			count[labels[i]] += 1
	new_labels, new_pngs = np.array(new_labels), np.array(new_pngs)
	new_labels, new_pngs = shuffle(new_labels, new_pngs, random_state=1)
	unique, counts = np.unique(new_labels, return_counts=True)
	new_dist = dict(zip(unique, counts))
	split = int(len(new_labels)*per)
	# import pdb;pdb.set_trace()
	# x_train, x_test, y_train, y_test
	return new_pngs[:split, :], new_pngs[split:, :], new_labels[:split], new_labels[split:]

def testing(label_set, dist):
	loc = './zach_cont_gest_1/*.png'
	# loc = './tmp/*.png'

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
			index = int(regex.findall(img)[1]) # index depends on how many int inside the path
			labels.append(label_set[index][0])
			image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
			image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
			pngs.append(image)
		pickle.dump(pngs, open("pngs.pkl", 'wb'))
		pickle.dump(labels, open("labels.pkl", 'wb'))
		pickle.dump(dist, open("dist.pkl", 'wb'))
	print "Reading time:", datetime.datetime.now()-t0
	labels = np.array(labels)
	# plot(labels, dist)

	pngs = np.array(pngs)
	length = pngs.shape[0]
	split = int(round(length*0.7))
	pngs = pngs.reshape((length, -1))
	x_train, x_test, y_train, y_test = spliting(pngs, labels, dist)
	t0 = datetime.datetime.now()
	classifier = svm.SVC(random_state=0) #gamma=0.001, degree=3, kernel='poly','rbf','linear'
	classifier.fit(x_train, y_train)
	print "Training time:", datetime.datetime.now()-t0

	t0 = datetime.datetime.now()
	expected = y_test
	predicted = classifier.predict(x_test)
	scoreI = classifier.score(x_train, y_train)
	scoreO = classifier.score(x_test, y_test)
	print "Testing time:", datetime.datetime.now()-t0
	print("\nTraining Accuracy: {:.2%}".format(scoreI))
	print("Testing Accuracy: {:.2%}\n".format(scoreO))
	print("Classification report for classifier %s:\n%s\n"
	      % (classifier, metrics.classification_report(expected, predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


if __name__ == "__main__":
	# save_img()
	label_set, dist = read_data()
	# png = load_img()
	testing(label_set, dist)