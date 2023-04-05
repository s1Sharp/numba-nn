from mnist import MNIST
import numpy as np

class IDataset(object):
    train_data = None
    test_data = None

    train_labels = None
    test_labels = None

    train_targets = None

    def __init__(self):
        raise NotImplementedError

    def get_train(self):
        raise NotImplementedError

    def get_test(self):
        raise NotImplementedError


class MnistDataSet(IDataset):
    NUMBER_OF_CLASSES = 10

    def __init__(self, ds_path="./mnist_dataset"):
        """Loads dataset of training and testing data"""

        print("Loading MNIST dataset ...", end='')
        # loading data
        self.train_data, self.train_labels = MNIST(ds_path).load_training()
        self.test_data, self.test_labels = MNIST(ds_path).load_testing()
        print(' ...', end='')

        # translate data into range (0,1)
        self.train_data = np.matrix(self.train_data, dtype=np.float64).T / 255
        self.test_data = np.matrix(self.test_data, dtype=np.float64).T / 255

        print(' ...', end='')
        self.train_labels = np.matrix(self.train_labels)
        self.test_labels = np.matrix(self.test_labels)

        # one hot encoding step
        self.train_targets = np.zeros((MnistDataSet.NUMBER_OF_CLASSES,
                                       self.train_data.shape[1]))
        for i in range(self.train_targets.shape[1]):
            self.train_targets[self.train_labels[0, i], i] = 1
        self.dataset_loaded = True
        print(" Done")

class FashionDataSet(IDataset):
    NUMBER_OF_CLASSES = 10

    def __init__(self, ds_path="./fashion_dataset"):
        """Loads fashion MNIST dataset of training and testing data"""

        print("Loading fashion MNIST dataset ...", end='')
        self.train_data, self.train_labels = MNIST(ds_path).load_training()
        self.test_data, self.test_labels = MNIST(ds_path).load_testing()
        print(' ...', end='')

        self.train_data = np.matrix(self.train_data, dtype=np.float64).T / 255
        self.test_data = np.matrix(self.test_data, dtype=np.float64).T / 255
        # self.train_data = np.matrix([[i / 255 for i in image]
        #                              for image in self.train_data]).T
        # self.test_data = np.matrix([[i / 255 for i in image]
        #                             for image in self.test_data]).T
        print(' ...', end='')
        self.train_labels = np.matrix(self.train_labels)
        self.test_labels = np.matrix(self.test_labels)

        self.train_targets = np.zeros((FashionDataSet.NUMBER_OF_CLASSES,
                                       self.train_data.shape[1]))
        for i in range(self.train_targets.shape[1]):
            self.train_targets[self.train_labels[0, i], i] = 1
        self.dataset_loaded = True
        print(" Done")