# cuda_nn_mnist

[Diploma project](https://kpfu.ru/student_diplom/10.160.178.20_GZJU1__O90UU3V_UXKIQTG1JB419O6X29AT5JMJE_M8JMK5MKZ_Carkov_MV.pdf) by Tsarkov Maksim KPFU 2022


FC Neural network implementation for digit and fashion classification powered by CUDA [numba](https://github.com/numba/numba).

## Resources
The library was built to work with [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and  [fashion MNIST](https://github.com/zalandoresearch/fashion-mnist).

## Hardware requirements
The main library is [Numba](https://numba.readthedocs.io/en/stable/cuda/index.html).
See the list of [CUDA-enabled GPU cards](https://developer.nvidia.com/cuda-gpus).

---

## How to run with docker

Docker:

    docker build -f ./Dockerfile -t cuda_app .
    docker run --gpus all cuda_app
    # "than attach to container with vscode"

Docker-compose:

    docker-compose up -d --build
    # "than check container logs"

# Log output example
```
Loading MNIST dataset ... ... ... Done
Epoch: 1 / 5 [####################] 25.3s, loss: 0.153804
learning rate: 0.15000
Train accuracy: 73.36%
Test accuracy: 74.2%
mean epoch loss: 0.17217

Epoch: 2 / 5 [####################] 14.3s, loss: 0.07089
learning rate: 0.14345
Train accuracy: 90.13%
Test accuracy: 90.39%
mean epoch loss: 0.10917

Epoch: 3 / 5 [####################] 12.0s, loss: 0.046125
learning rate: 0.13120
Train accuracy: 92.17%
Test accuracy: 92.1%
mean epoch loss: 0.05291

Epoch: 4 / 5 [####################] 12.5s, loss: 0.053371
learning rate: 0.11476
Train accuracy: 89.46%
Test accuracy: 89.61%
mean epoch loss: 0.05102

Epoch: 5 / 5 [####################] 11.4s, loss: 0.033853
learning rate: 0.09600
Train accuracy: 94.12%
Test accuracy: 94.09%
mean epoch loss: 0.04286

[[0.08378762 0.08542943]
 [0.08735926 0.08799926]
 [0.10407113 0.0854538 ]
 [0.09083512 0.20800408]
 [0.08528696 0.08618262]
 [0.0838901  0.10128012]
 [0.08387205 0.08634175]
 [0.20787338 0.08542049]
 [0.08453341 0.08815595]
 [0.08849096 0.08573251]]
[[7 3]]
```

---

## How to set up environment 

Linux:

    python3 -m venv env
    source env/bin/activate
    python3 -m pip install -r requirements.txt

Windows (manually):

    python -m venv env
    .\env\Scripts\activate.bat
    python -m pip install -r requirements.txt

Windows (automaticly):

    .\scripts\install_py_env.ps1

For deactivate the Python venv:

    deactivate

### Update requirements

    pip freeze > requirements.txt
