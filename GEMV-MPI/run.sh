#!/bin/bash

executable=$1
processes=(2 3 4 5 6 7 8 9)
echo "$executable"
mpiexec -np 1 "$executable" > "${executable}_output.txt"
for proc in "${processes[@]}"; do
  mpiexec -np "$proc" "$executable" >> "${executable}_output.txt"
done