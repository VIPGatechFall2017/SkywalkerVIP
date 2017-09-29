import numpy as np
import pandas as pd
from sklearn import svm, metrics
from skimage import io
import re
import glob
import cv2
import datetime

def read_data():
	df = pd.read_csv('data.txt', sep=';', header=None, delimiter=' ')
	df = df.drop(0, axis=1)
	df[5] = df[5].map(lambda x: x.rstrip(';'))
	df[5] = df[5].map(lambda x: int(x))
	data = np.array(df)
	data = np.logical_and(data, 1)
	return data

def save_img():
	loc = './zach_img/*.png'
	png = []
	count = 0
	for img in glob.glob(loc):
		print count
		png.append(io.imread(img))
		count+=1
	np.save('imgs.npy', png)

def load_img():
	return np.load('imgs.npy')

def testing(label_set):
	loc = '../../zach_img/*.png'
	# loc = './tmp/*.png'

	addrs = glob.glob(loc)
	regex = re.compile(r'\d+')
	labels = []
	pngs = []
	tmp_label = []

	t0 = datetime.datetime.now()
	for img in addrs:
		index = int(regex.findall(img)[0])
		labels.append(label_set[index][2])
		image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_CUBIC)
		pngs.append(image)
	print "Reading time:", datetime.datetime.now()-t0

	pngs = np.array(pngs)
	length = pngs.shape[0]
	split = int(round(length*0.7))
	pngs = pngs.reshape((length, -1))
	train_pngs = pngs[:split, :]
	test_pngs = pngs[split:, :]
	train_labels = labels[:split]
	test_labels = labels[split:]

	t0 = datetime.datetime.now()
	classifier = svm.SVC(gamma=0.001)
	classifier.fit(train_pngs, train_labels)
	print "Training time:", datetime.datetime.now()-t0

	t0 = datetime.datetime.now()
	expected = test_labels
	predicted = classifier.predict(test_pngs)
	print "Testing time:", datetime.datetime.now()-t0

	print("Classification report for classifier %s:\n%s\n"
	      % (classifier, metrics.classification_report(expected, predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


if __name__ == "__main__":
	# save_img()
	label_set = read_data()
	# png = load_img()
	# print png[0]
	testing(label_set)