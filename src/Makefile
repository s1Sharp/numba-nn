all:
	source ./env/bin/activate && python nn.py
$(filename):
	source ./env/bin/activate && python $(filename).py

prof:
	nvprof -o nvprof%p.out --profile-child-processes  --unified-memory-profiling off ./env/Scripts/python.exe ./nn.py

vis_prof:
	nvvp --profile-child-processes  --unified-memory-profiling off ./env/Scripts/python.exe ./nn.py

c_prof:
	python -m cProfile -o output.pstats nn.py gprof2dot.py -f pstats output.pstats | dot -Tpng -o output.png

CUDA Version 11.6.0

Container image Copyright (c) 2016-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use the NVIDIA Container Toolkit to start this container with GPU support; see
   https://docs.nvidia.com/datacenter/cloud-native/ .

*************************
** DEPRECATION NOTICE! **
*************************
THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
    https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md

Traceback (most recent call last):
  File "nn.py", line 289, in <module>
    mnist_dataset = MnistDataSet()
  File "/cnn/mnist_loader.py", line 31, in __init__
    self.train_data, self.train_labels = MNIST(ds_path).load_training()
  File "/usr/local/lib/python3.8/dist-packages/mnist/loader.py", line 125, in load_training
    ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
  File "/usr/local/lib/python3.8/dist-packages/mnist/loader.py", line 247, in load
    with self.opener(path_lbl, 'rb') as file:
  File "/usr/local/lib/python3.8/dist-packages/mnist/loader.py", line 239, in opener
    return open(path_fn, *args, **kwargs)
FileNotFoundError: [Errno 2] No such file or directory: './mnist_dataset/train-labels-idx1-ubyte'