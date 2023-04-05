#!/bin/bash

if [ -d ./mnist_dataset ]; then
  echo "data mnist_dataset directory already present, exiting"
else
  mkdir mnist_dataset
  wget --recursive --level=1 --cut-dirs=3 --no-host-directories \
  --directory-prefix=mnist_dataset --accept '*.gz' http://yann.lecun.com/exdb/mnist/
  pushd mnist_dataset
  gunzip *
  popd
fi


if [ -d ./fashion_dataset ]; then
  echo "data fashion_dataset directory already present, exiting"
else
  mkdir fashion_dataset
  wget --directory-prefix=fashion_dataset --accept '*.gz' https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz
  wget --directory-prefix=fashion_dataset --accept '*.gz' https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz
  wget --directory-prefix=fashion_dataset --accept '*.gz' https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz
  wget --directory-prefix=fashion_dataset --accept '*.gz' https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz
  pushd fashion_dataset
  gunzip *
  popd
fi