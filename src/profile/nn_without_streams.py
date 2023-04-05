import warnings
from math import exp, tanh, ceil
from time import time

import numpy as np
from numba import cuda, core

import cuda_kernels as cu_k
from mnist_loader import *

class NeuralNetwork:
    sigmoid = (lambda x: 1 / (1 + exp(-x)), lambda y: y * (1 - y))
    tanh = (lambda x: tanh(x), lambda y: 1 - (y ** 2))

    def __init__(self, in_features, hidden_layer_sizes, out_features, dataset: IDataset, thread_per_block=(8, 8), act_f="sigmoid"):
        
        # Threads per block for cuda
        self.tpb = thread_per_block

        self.in_features = in_features
        self.hidden_layer_sizes = hidden_layer_sizes
        self.out_features = out_features
        
        
        self.__init_dataset(dataset)
        self.__init_act_func(act_f)
        # self.__init_fast_mat_mul()


        scale1 = np.sqrt(1 / in_features)
        scale2 = np.sqrt(1 / hidden_layer_sizes)
        # np.random.uniform(low=-scale, high=scale, size=shape)
        self.weights = [
            cuda.to_device(np.random.uniform(low=-scale1, high=scale1, size=(hidden_layer_sizes, in_features))),
            cuda.to_device(np.random.uniform(low=-scale2, high=scale2, size=(out_features, hidden_layer_sizes)))
            ]

        self.biases = [
            cuda.to_device(np.random.uniform(low=-scale1, high=scale1, size=(hidden_layer_sizes, 1))),
            cuda.to_device(np.random.uniform(low=-scale2, high=scale2, size=(out_features, 1)))
            ]
        # self.weights = [
        #     cuda.to_device(np.random.rand(hid_nodes, in_nodes) * 2 - 1),
        #     cuda.to_device(np.random.rand(out_nodes, hid_nodes) * 2 - 1)]

        # self.biases = [
        #     cuda.to_device(np.random.rand(hid_nodes, 1) * 2 - 1),
        #     cuda.to_device(np.random.rand(out_nodes, 1) * 2 - 1)]
    
    def __init_fast_mat_mul(self):
        cu_k.matmul = cu_k.fast_matmul

    def __init_act_func(self, act_f):
        act_list = {'sigmoid': { 
                        'act':cu_k.sigmoid, 'dact':cu_k.dsigmoid 
                            },
                    'relu': { 
                        'act':cu_k.relu, 'dact':cu_k.drelu 
                        }
                    }
        assert (act_f in act_list)

        cu_k.feedforward_step = cu_k.make_feedforward_step(act_list[act_f]['act'])
        cu_k.gradient = cu_k.make_gradient(act_list[act_f]['dact'])

    def __init_dataset(self, dataset: IDataset):
        
        # Each column of data is a single array of input
        # (e.g. a single image from MNIST)
        self.train_data = dataset.train_data # None
        self.test_data = dataset.test_data # None

        # Labels for corresponding data
        self.train_labels = dataset.train_labels # None
        self.test_labels = dataset.test_labels # None

        # Desired output for corresponding data based on self.train_labels
        self.train_targets = dataset.train_targets # None
        
        self.dataset_loaded = False

        return

    def get_grid_dim(self, out_shape):
        """Returns cuda grid dimensions needed for kernel execution"""
        return (ceil(out_shape[0] / self.tpb[0]),
                ceil(out_shape[1] / self.tpb[1])), self.tpb

    def feedforward_cuda(self, inputs):
        """Returns outputs of NN for each input"""

        inputs_d = cuda.to_device(inputs)
        outputs_d = cuda.device_array((self.weights[0].shape[0],
                                       inputs.shape[1]))

        cu_k.feedforward_step[self.get_grid_dim(outputs_d.shape)](inputs_d, self.weights[0], self.biases[0], outputs_d)
        cuda.synchronize()

        inputs_d = outputs_d
        outputs_d = cuda.device_array((self.weights[1].shape[0],
                                       inputs.shape[1]))

        cu_k.feedforward_step[self.get_grid_dim(outputs_d.shape)](inputs_d, self.weights[1], self.biases[1], outputs_d)
        cuda.synchronize()

        softmax_d = cuda.device_array((outputs_d.shape[0], outputs_d.shape[1]))
        cu_k.softmax[self.get_grid_dim(outputs_d.shape)](outputs_d, softmax_d)

        return softmax_d.copy_to_host()

    def test_accuracy(self):
        """Returns categorical accuracy of NN on self.test_data"""

        output = self.feedforward_cuda(self.test_data)
        return (output.argmax(axis=0) == self.test_labels[0, :]).sum() / self.test_data.shape[1] * 100

    def train_accuracy(self):
        """Returns categorical accuracy of NN on self.train_data"""

        output = self.feedforward_cuda(self.train_data)
        return (output.argmax(axis=0) == self.train_labels[0, :]).sum() / self.train_data.shape[1] * 100

    def print_accuracy(self):
        train_acc = self.train_accuracy()
        test_acc = self.test_accuracy()
        
        print(f"Train accuracy: {round(train_acc, 2)}%")
        print(f"Test accuracy: {round(test_acc, 2)}%")

        return train_acc, test_acc

    def backpropogation_cuda(self, inputs, targets, lr, batch,
                            stream1, stream2):
        """
        Implements batch gradient descent.
        Each column of targets is desired output of NN for each input
        """
        # copy data to device
        inputs_d = cuda.to_device(inputs) # 784 x batch
        targets_d = cuda.to_device(targets) # 10 x batch

        hidden_d = cuda.device_array((self.hidden_layer_sizes, batch)) # hidden_layer_sizes x batch

        # forward from IN layer to HIDDEN layer
        # IN --> HIDDEN
        cu_k.feedforward_step[self.get_grid_dim(hidden_d.shape)](inputs_d, self.weights[0], self.biases[0], hidden_d)
        cuda.synchronize()

        # set output from HIDDEN layer to input for OUT layer
        inputs_d = hidden_d
        outputs_d = cuda.device_array((self.out_features, batch))
        
        # forward from HIDDEN layer to OUT layer
        # HIDDEN --> OUT
        cu_k.feedforward_step[self.get_grid_dim(outputs_d.shape)](inputs_d, self.weights[1], self.biases[1], outputs_d)
        cuda.synchronize()

        # create device array tensor
        # 10 x batch
        errors_d = cuda.device_array(targets.shape)

        # create tensor for HIDDEN layer derivative
        hidden_errors_d = cuda.device_array((self.hidden_layer_sizes, batch))
        
        # == > backward step / weights update step < ==
        
        # compute error
        cu_k.subtract[self.get_grid_dim(errors_d.shape)](targets_d, outputs_d, errors_d)
        cuda.synchronize()
        errors_h = errors_d

        # self.weights[1].T
        # compute HIDDER error
        cu_k.matmul_T[self.get_grid_dim(hidden_errors_d.shape)](self.weights[1], errors_d, hidden_errors_d)
        cuda.synchronize()

        # gradient for HIDDEN
        gradient_d = cuda.device_array(outputs_d.shape)
        # compute gradient for HIDDEN layer
        cu_k.gradient[self.get_grid_dim(gradient_d.shape)](outputs_d, errors_d, lr, gradient_d)
        cuda.synchronize()

        # delta for HIDDEN layer bias
        delta_b_d = cuda.device_array(self.biases[1].shape)
        # compute delta for HIDDEN layer bias by gradient
        cu_k.sum_cols[ceil(self.biases[1].shape[0] / 4), 4](gradient_d, delta_b_d)
        cuda.synchronize()

        # update HIDDEN layer bias by delta
        cu_k.add[self.get_grid_dim(self.biases[1].shape)](self.biases[1], delta_b_d)


        # delta for HIDDEN layer weights
        delta_w_d = cuda.device_array((gradient_d.shape[0], hidden_d.shape[0]))
        # hidden_d.T
        # compute delta for HIDDEN layer weights by gradient
        cu_k.matmul_T[self.get_grid_dim(self.weights[1].shape)](gradient_d, hidden_d, delta_w_d)
        cuda.synchronize()

        # update HIDDEN layer weights by delta
        cu_k.add[self.get_grid_dim(self.weights[1].shape)](self.weights[1], delta_w_d)

        # gradient for IN layer
        gradient_d = cuda.device_array(hidden_d.shape)
        # compute gradient for IN layer
        cu_k.gradient[self.get_grid_dim(gradient_d.shape)](hidden_d, hidden_errors_d, lr, gradient_d)
        cuda.synchronize()

        # delta for IN layer bias
        delta_b_d = cuda.device_array(self.biases[0].shape)
        # compute delta for IN layer bias by gradient
        cu_k.sum_cols[ceil(self.biases[0].shape[0] / 4), 4](gradient_d, delta_b_d)
        cuda.synchronize()
        # update IN layer bias by delta
        cu_k.add[self.get_grid_dim(self.biases[0].shape)](self.biases[0], delta_b_d)

        # delta for IN layer weights
        delta_w_d = cuda.device_array((gradient_d.shape[0], inputs.shape[0]))
        # W * x.T
        inputs_d = cuda.to_device(inputs.T)
        # delta for IN layer weights
        cu_k.matmul[self.get_grid_dim(self.weights[0].shape)](gradient_d, inputs_d, delta_w_d)
        cuda.synchronize()

        # update IN layer weights by delta
        cu_k.add[self.get_grid_dim(self.weights[0].shape)](self.weights[0], delta_w_d)
        cuda.synchronize()

        return errors_h.copy_to_host()

    def train(self, epochs: int, learning_rate: float, batch_size: int):
        """
        Train neural network on a dataset, divided into batches.
        Weights are corrected after each batch
        """
        stream1 = cuda.stream()
        stream2 = cuda.stream()


        lr = learning_rate
        for r in range(epochs):
            start_time = time()

            # lr = lr * 9e-1 ** (r / epochs)
            lr = lr * exp(-r / epochs)
            perm = np.random.permutation(self.train_targets.shape[1])
            batch_count = ceil(perm.size / batch_size)
            for i in range(batch_count):
                pr_bar = ('#' * ((i + 1) * 20 // batch_count)).ljust(20, ' ')

                batch_perm = perm[i * batch_size: (i + 1) * batch_size]

                with cuda.profiling():
                    loss = self.backpropogation_cuda(self.train_data[:, batch_perm],
                                                    self.train_targets[:, batch_perm],
                                                    lr,
                                                    batch_size,
                                                    stream1, stream2)

                # return

                dur = round(time() - start_time, 1)
                loss = np.mean(loss ** 2)
                print(f"\rEpoch: {r + 1} / {epochs} [{pr_bar}] {dur}s, loss: {loss:.5f}", end='')
            print()
            print(f'learning rate: {lr:.5f}')
            self.print_accuracy()
            print()

    def save_weights(self, filename: str):
        """Save weights to weights directory as an .npz archive"""
        np.savez_compressed(f"weights/{filename}",
                            *[self.weights[i].copy_to_host()
                              for i in range(len(self.weights))],
                            *[self.biases[i].copy_to_host()
                              for i in range(len(self.biases))])

    def load_weights(self, filename: str):
        """Load weights stored as an .npz archive from weights directory"""
        loaded = np.load(f"weights/{filename}.npz")
        self.weights = [loaded['arr_0'], loaded['arr_1']]
        self.biases = [loaded['arr_2'], loaded['arr_3']]


if __name__ == '__main__':
    # Silence Numba warnings about low occupancy of GPU
    warnings.simplefilter('ignore',
                          category=core.errors.NumbaPerformanceWarning)

    mnist_dataset = MnistDataSet()
    # fashion_dataset = FashionDataSet()

    dc = NeuralNetwork(784, 256, 10, mnist_dataset, act_f='relu')

    dc.train(1, 0.15, 100)
    # dc.save_weights('weights1')
    print(dc.feedforward_cuda(mnist_dataset.test_data[:, 111:113]))
    print(mnist_dataset.test_labels[:,111:113])
