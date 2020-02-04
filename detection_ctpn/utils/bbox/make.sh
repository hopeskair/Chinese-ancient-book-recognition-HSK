#!/usr/bin/env bash

python setup.py install
mv build/*/*.so ./
rm -rf build/
rm *.c