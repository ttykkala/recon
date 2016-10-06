#!/bin/sh

cd extlibs
cmake -DINSTALL_ROOT=$(pwd) CMakeLists.txt
make
make install
cd ..
