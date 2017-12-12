import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

# reference : https://github.com/tensorflow/tensorflow/blob/7c36309c37b04843030664cdc64aca2bb7d6ecaa/tensorflow/contrib/learn/python/learn/datasets/mnist.py#L160
class DataSet(object):

    def __init__(self, images, labels, labels_1h=None, dtype=dtypes.float32, reshape=True):
        if reshape:
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._labels_1h = labels_1h
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def labels_1h(self):
        return self._labels_1h

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            self._labels_1h = self._labels_1h[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end], self._labels_1h[start:end]

def read_data_sets(images, labels, 
                   train_size=0.6, val_size=0.2, test_size=0.2,
                   dtype=dtypes.float32,
                   reshape=False):

    sample_size = labels.shape[0]
    validation_size = int(sample_size * val_size)
    train_size = int(sample_size * train_size)

    classes = np.array(list(set(labels)))
    labels_1h = np.zeros((sample_size, classes.shape[0]))
    for i in range(0, sample_size):
        labels_1h[i] = ((labels[i]==classes).astype(int))

    val_images = images[:validation_size]
    val_labels = labels[:validation_size]
    val_labels_1h = labels_1h[:validation_size]

    train_images = images[validation_size:validation_size+train_size]
    train_labels = labels[validation_size:validation_size+train_size]
    train_labels_1h = labels_1h[validation_size:validation_size+train_size]

    test_images = images[validation_size+train_size:]
    test_labels = labels[validation_size+train_size:]
    test_labels_1h = labels_1h[validation_size+train_size:] 

    train = DataSet(train_images, train_labels, labels_1h=train_labels_1h, dtype=dtype, reshape=reshape)
    val = DataSet(val_images, val_labels, labels_1h=val_labels_1h, dtype=dtype, reshape=reshape)
    test = DataSet(test_images, test_labels, labels_1h=test_labels_1h, dtype=dtype, reshape=reshape)

    return base.Datasets(train=train, validation=val, test=test)