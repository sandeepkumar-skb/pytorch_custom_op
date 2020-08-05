#!/bin/bash

rm -rf build;
mkdir build;
cd build

cmake -DCMAKE_PREFIX_PATH=/home/sandeep/anaconda3/lib/python3.7/site-packages/torch ..

make -j$(nproc)
