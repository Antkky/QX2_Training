#!/bin/bash

echo "Generating"
cd build
cmake ..

echo "Compiling"
cmake --build . --config release

echo "Running"
cd ../
./bin/QX2_Training ./data/processed/tData.csv
