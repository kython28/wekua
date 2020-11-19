#!/usr/bin/bash

make
sudo make install
icc -g -lOpenCL -lwekua -O2 probe.c -o probe -lm
#valgrind ./probe
./probe