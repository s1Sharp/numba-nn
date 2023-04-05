import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sig_derivative(x, step):
    return (sigmoid(x+step) - sigmoid(x)) / step


def show_sigmoid():
    x = np.linspace(-10, 10, 1000)

    y1 = sigmoid(x)
    y2 = sig_derivative(x, 0.0000000000001)

    plt.plot(x, y1, label=r'sigmoid, $\sigma(x) = \frac{1}{1+e^{-x}}$')
    plt.plot(x, y2, label='derivative, $\sigma(x)\' = 1\cdot(1-x)$')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title("Sigmoid")
    plt.show()

def show_relu():
    xd = np.linspace(-10, 10, 1000)

    y1 = list(map(lambda x: 1. + 0.01 * (x - 1.) if x > 1 else x * 0.01 if x < 0 else x, xd))
    y2 = list(map(lambda x: 0.01 if x < 0 or x > 1 else 1, xd))


    plt.plot(xd, y1, label='LeakyReLu, $(1.5)$')
    plt.plot(xd, y2, label='derivative, $(1.6)$')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title("LeakyReLu")
    plt.show()


show_sigmoid()
show_relu()