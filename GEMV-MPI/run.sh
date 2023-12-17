#!/bin/bash

# Change this list of processes number
processes=(2 3 4 5 6 7 8)

executable=$1
log_file="${executable}_log_$2x$3.txt"
log_file="${log_file//.exe/}"
times_file="${executable}_times_$2x$3.txt"
times_file="${times_file//.exe/}"

echo "Run in 1 process..."
mpiexec -np 1 "$executable" "$2" "$3" "${times_file}" > "${log_file}"

for proc in "${processes[@]}"; do
  echo "Run in ${proc} processes..."
  mpiexec -np "$proc" "$executable" "$2" "$3" "${times_file}" >> "${log_file}"
done
