from math import exp, log
from numba import cuda, float64

"""Cuda kernels for neural network"""

@cuda.jit(device=True)
def matmul_device(matrix1, matrix2):
    i, batch = cuda.grid(2)
    if i < matrix1.shape[0] and batch < matrix2.shape[1]:
        tmp_value = 0.
        for j in range(matrix1.shape[1]):
            tmp_value += matrix1[i, j] * matrix2[j, batch]
        return tmp_value


@cuda.jit
def matmul(matrix1, matrix2, out):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        tmp_value = 0.
        for k in range(matrix1.shape[1]):
            tmp_value += matrix1[i, k] * matrix2[k, j]
        out[i, j] = tmp_value


@cuda.jit
def matmul_T(matrix1_T, matrix2, matrix_out):
    i, j = cuda.grid(2)
    if i < matrix_out.shape[0] and j < matrix_out.shape[1]:
        tmp_value = 0.
        for k in range(matrix1_T.shape[0]):
            tmp_value += matrix1_T[k, i] * matrix2[k, j]
        matrix_out[i, j] = tmp_value


@cuda.jit
def softmax(z, out):
    x, batch = cuda.grid(2)
    if x < z.shape[0] and batch < z.shape[1]:
        sum_exp = 0 
        for k in range(z.shape[0]):
            sum_exp += exp(z[k, batch])
        out[x, batch] = exp(z[x, batch]) / sum_exp


@cuda.jit
def dsoftmax(z, out):
    x, batch = cuda.grid(2)
    if x < z.shape[0] and batch < z.shape[1]:
        sum_exp = 0 
        for k in range(z.shape[0]):
            sum_exp += exp(z[k, batch])
        out[x, batch] = exp(z[x, batch]) / sum_exp


@cuda.jit
def gradient_sm(outputs, err, lr, gradients):
    x, batch = cuda.grid(2)
    if x < outputs.shape[0] and batch < outputs.shape[1]:
        gradients[x, batch] = dsoftmax(outputs[x, batch]) * err[x, batch] * lr


@cuda.jit(device=True)
def relu(x):
    '''LeakyReLU'''
    return 1. + 0.01 * (x - 1.) if x > 1 else x * 0.01 if x < 0 else x


@cuda.jit(device=True)
def drelu(y):
    '''derivative LeakyReLU'''
    return 0.01 if y < 0 or y > 1 else 0.15


@cuda.jit(device=True)
def sigmoid(x):
    '''sigmoid'''
    return 1 / (1 + exp(-x))


@cuda.jit(device=True)
def dsigmoid(y):
    '''derivative sigmoid'''
    return y * (1 - y)


@cuda.jit
def feedforward_step(inputs, weights, biases, outputs):
    x, batch = cuda.grid(2)
    if x < outputs.shape[0] and batch < outputs.shape[1]:
        outputs[x, batch] = matmul_device(weights, inputs)
        cuda.atomic.add(outputs, (x, batch), biases[x, 0])
        outputs[x, batch] = sigmoid(outputs[x, batch])


@cuda.jit
def gradient(outputs, err, lr, gradients):
    x, batch = cuda.grid(2)
    if x < outputs.shape[0] and batch < outputs.shape[1]:
        gradients[x, batch] = dsigmoid(outputs[x, batch]) * err[x, batch] * lr


@cuda.jit
def sum_cols(in_arr, out):
    x = cuda.grid(1)
    if x < in_arr.shape[0]:
        out[x, 0] = 0
        for y in range(in_arr.shape[1]):
            out[x, 0] +=  in_arr[x, y]
            # cuda.atomic.add(out, (x, 0), in_arr[x, y])


@cuda.jit
def subtract(arr1, arr2, out):
    x, y = cuda.grid(2)
    if x < arr1.shape[0] and y < arr1.shape[1]:
        out[x, y] = arr1[x, y] - arr2[x, y]


@cuda.jit
def add(arr1, arr2):
    x, y = cuda.grid(2)
    if x < arr1.shape[0] and y < arr1.shape[1]:
        arr1[x, y] += arr2[x, y]
        # cuda.atomic.add(arr1, (x, y), arr2[x, y])


@cuda.jit
def mean_loss(arr1, arr2, out_):
    x, y = cuda.grid(2)
    if x < arr1.shape[0] and y < arr1.shape[1]:
        out[x, y] = (arr1[x, y] - arr2[x, y]) ** 2


@cuda.jit
def cross_enropy_loss(target, data_out, loss):
    x, y = cuda.grid(2)
    if x < target.shape[0] and y < target.shape[1]:
        loss[x, y] = -log(data_out[x, y]) if target[x, y] == 1 else -log(1-data_out[x, y])


def make_feedforward_step(act_f):   
    @cuda.jit
    def _feedforward_step(inputs, weights, biases, outputs):
        x, batch = cuda.grid(2)
        if x < outputs.shape[0] and batch < outputs.shape[1]:
            outputs[x, batch] = matmul_device(weights, inputs)
            cuda.atomic.add(outputs, (x, batch), biases[x, 0])
            outputs[x, batch] = act_f(outputs[x, batch])
    return _feedforward_step


def make_gradient(dact_f):
    @cuda.jit
    def _gradient(outputs, err, lr, gradients):
        x, batch = cuda.grid(2)
        if x < outputs.shape[0] and batch < outputs.shape[1]:
            gradients[x, batch] = dact_f(outputs[x, batch]) * err[x, batch] * lr
    return _gradient


TPB = 32
@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float64)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp