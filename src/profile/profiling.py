import cuda_kernels as cu_k
import numpy as np
from numba import cuda, core
from math import ceil
from time import time
import warnings
import pandas as pd
import matplotlib.pyplot as plt

tpb = (16, 16)
warnings.simplefilter('ignore',
                        category=core.errors.NumbaPerformanceWarning)

def get_grid_dim(out_shape):
    """Returns cuda grid dimensions needed for kernel execution"""
    return (max(256,ceil(out_shape[0] / tpb[0])),
            max(256,ceil(out_shape[1] / tpb[1]))), tpb

def test_event_elapsed(n):

        size = (n , n)
        data1 = np.random.random(size)
        data2 = np.random.random(size)

        data1_d = cuda.device_array(size, dtype=np.double)
        data2_d = cuda.device_array(size, dtype=np.double)
        dataout_d = cuda.device_array(size, dtype=np.double)
        cuda.to_device(data1, to=data1_d)
        cuda.to_device(data2, to=data2_d)

        evtstart = cuda.event()
        evtend = cuda.event()

        evtstart.record()
        # GPU
        cu_k.matmul[get_grid_dim(size)](data1_d, data2_d, dataout_d)

        evtend.record()
        evtend.wait()
        evtend.synchronize()

        # CPU
        start_time = time()
        tmp = np.matmul(data1, data2)
        cpu_time = time() - start_time
        gpu_time = evtstart.elapsed_time(evtend) / 1000  # from ms to sec
        # Exercise the code path
        
        print(f'M_size: {n}')
        print(f'GPU time: {gpu_time}')
        print(f'CPU time: {cpu_time}')
        return cpu_time, gpu_time

def show_df(path):
    df = pd.read_csv('bench/mat_mul.csv')

    y1 = df['CPU_time']
    y2 = df['GPU_time']
    x = df['matrix_size']

    plt.plot(x, y1, label='CPU time (AMD Athlon x4 870K)')
    plt.plot(x, y2, label='GPU time (NVIDIA GTX 1050TI')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title("Time to execute matrix mult operation on CPU and on GPU")
    plt.xlabel('matrix size')
    plt.ylabel('time, sec.')
    plt.savefig('bench/mat_mul.png')
    plt.show()

def show_df2(path):
    df = pd.read_csv('bench/mat_mul.csv')

    y1 = df['CPU_time']
    y2 = df['GPU_time']
    x = df['matrix_size']

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(x, y1, label='CPU time (AMD Athlon x4 870K)')
    ax1.plot(x, y2, label='GPU time (NVIDIA GTX 1050TI')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_title("Time to execute matrix mult operation on CPU and on GPU")
    ax1.set_xlabel('matrix size')
    ax1.set_ylabel('time, sec.')


    ax2.plot(x, (y1/y2), label='execution time ratio CPU / GPU')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.set_xlabel('matrix size')
    ax2.set_ylabel('ratio')


    plt.savefig('bench/mat_mul.png')
    plt.show()

def show_df3(path):
    df = pd.read_csv('bench/sum_cols.csv')

    y1 = df['CPU_time']
    y2 = df['GPU_time']
    x = df['matrix_size']

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(x, y1, label='CPU time (AMD Athlon x4 870K)')
    ax1.plot(x, y2, label='GPU time (NVIDIA GTX 1050TI')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_title("Time to execute matrix sum operation on CPU and on GPU")
    ax1.set_xlabel('matrix size')
    ax1.set_ylabel('time, sec.')


    ax2.plot(x, (y1/y2), label='execution time ratio CPU / GPU')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.set_xlabel('matrix size')
    ax2.set_ylabel('ratio')


    # plt.savefig('bench/sum_cols.png')
    plt.show()


def test_sum_cols(n):

        size = (n , n)
        data1 = np.random.random(size)
        data2 = np.random.random(size)

        data1_d = cuda.device_array(size, dtype=np.double)
        data2_d = cuda.device_array(size, dtype=np.double)
        cuda.to_device(data1, to=data1_d)
        cuda.to_device(data2, to=data2_d)

        evtstart = cuda.event()
        evtend = cuda.event()

        evtstart.record()
        # GPU
        cu_k.add[get_grid_dim(size)](data1_d, data2_d)

        evtend.record()
        evtend.wait()
        evtend.synchronize()

        # CPU
        start_time = time()
        tmp = np.sum(data1, data2)
        cpu_time = time() - start_time
        gpu_time = evtstart.elapsed_time(evtend) / 1000  # from ms to sec
        # Exercise the code path
        
        print(f'M_size: {n}')
        print(f'GPU time: {gpu_time}')
        print(f'CPU time: {cpu_time}')
        return cpu_time, gpu_time

def bench_sum_cols():
        # _compile func code
        cpu_t, gpu_t = test_event_elapsed(10)

        CPU_TIME = []
        GPU_TIME = []
        M_SIZE = []

        for m_size in range(50, 8000, 50):
                cpu_t, gpu_t = test_event_elapsed(m_size)
                CPU_TIME.append(cpu_t)
                GPU_TIME.append(gpu_t)
                M_SIZE.append(m_size)
        
        df = pd.DataFrame({'CPU_time': CPU_TIME,
                   'GPU_time': GPU_TIME,
                   'matrix_size': M_SIZE})
        df.to_csv('bench/sum_cols.csv', index=False)

def bench_matmul():
        # _compile func code
        cpu_t, gpu_t = test_event_elapsed(10)

        CPU_TIME = []
        GPU_TIME = []
        M_SIZE = []

        for m_size in range(50, 8000, 50):
                cpu_t, gpu_t = test_event_elapsed(m_size)
                CPU_TIME.append(cpu_t)
                GPU_TIME.append(gpu_t)
                M_SIZE.append(m_size)
        
        df = pd.DataFrame({'CPU_time': CPU_TIME,
                   'GPU_time': GPU_TIME,
                   'matrix_size': M_SIZE})
        df.to_csv('bench/mat_mul.csv', index=False)

#bench_matmul()
# show_df2('bench/mat_mul.csv')
#bench_sum_cols()
#show_df3('bench/sum_cols.csv')

from mnist_loader import *
from nn import *
        # self.train_result = {'epoch':[],
        #                      'loss':[],
        #                      'acc': [] (train_acc, test_acc, r)
        #                      }
def plot_loss(res):
    l2 = 20
    epoch = np.array(res['epoch'][1:])
    loss = np.array(res['loss'][1:] ) ** (1/2)
    for idx in range(0, len(loss), l2):
        mloss = np.mean(loss[idx:idx+l2])
        loss[idx:idx+l2] = list(map(lambda _: mloss, loss[idx:idx+l2]))

    acc_tr = [elem[0] for elem in res['acc']]
    acc_t = [elem[1] for elem in res['acc']]
    nep = [elem[2] for elem in res['acc']]

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(epoch, loss, label='loss')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.set_title(f"Loss value by epoch")
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

# Set the width of the bars
    wd = 0.2
    bt = min(acc_t) - 3
    ax2.bar(np.array(nep)-wd/2, np.array(acc_tr) - bt, color='blue', label='Train accuracy', width=wd, bottom=bt)
    ax2.bar(np.array(nep)+wd/2, np.array(acc_t) - bt, color='red', label='Test accuracy', width=wd, bottom=bt)
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy, %')

    fig.suptitle(f"nentwork with hidden size {res['dim'][1]}, lr={res['lr'][0]}, batch={res['batch']}, act func: {res['af']}", fontsize=16)
    # plt.savefig('bench/sum_cols.png')
    plt.show()

def show_df5():
    df = pd.read_csv('bench/batch.csv')
    t = np.array(df['time'])
    b = np.array(df['batch'])
    ac = df['test_acc']

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(b, t, label='time')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.set_title(f"Time for one epoch")
    ax1.set_xlabel('batch')
    ax1.set_ylabel('time, sec.')

# Set the width of the bars
    wd = 10
    bt = 25
    ax2.plot(b, ac, label='Test accuracy') # , color='orange', label='Test accuracy', width=wd, bottom=bt)
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.set_xlabel('batch')
    ax2.set_ylabel('accuracy, %')

    fig.suptitle(f"Train time for one epoch by batch size", fontsize=16)
    # plt.savefig('bench/sum_cols.png')
    plt.show()

if __name__ == '__main__':
    # Silence Numba warnings about low occupancy of GPU
    warnings.simplefilter('ignore',
                          category=core.errors.NumbaPerformanceWarning)
#     show_df5()
#     exit(0)
#     mnist_dataset = MnistDataSet()
#     # fashion_dataset = FashionDataSet()
#     dc = NeuralNetwork(784, 512, 10, dataset=mnist_dataset, 
#                                 act_f='relu',               # relu sigmoid
#                                 thread_per_block=(32,32)) 
#     batch = 650
#     ti = dc.train(1, 0.1, batch)
#     print(ti,batch, sep=',')
#     exit(0)
    fashion_dataset = FashionDataSet()
    mnist_dataset = MnistDataSet()


    dc = NeuralNetwork(784, 512, 10, dataset=fashion_dataset, 
                                    act_f='sigmoid',               # relu sigmoid
                                    thread_per_block=(8,8)) 

    dc.train(20, 0.1, 100)
    res = dc.get_train_result()
    # plot_loss(res)
    dc2 = NeuralNetwork(784, 256, 10, dataset=fashion_dataset, 
                                    act_f='sigmoid',               # relu sigmoid
                                    thread_per_block=(8,8)) 

    dc2.train(20, 0.1, 100)
    res2 = dc2.get_train_result()

    plot_loss(res)
    plot_loss(res2)
    # for lr in [x/10000 for x in range(300, 1, -30)]:
    #     dc = NeuralNetwork(784, 256, 10)
