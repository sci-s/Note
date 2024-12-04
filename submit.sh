#!/bin/bash
#PBS -N ii
#PBS -q samdi2
#PBS -A etc
#PBS -l select=1:ncpus=32:mpiprocs=32:ompthreads=1
#PBS -l walltime=00:30:00

module purge
module load craype-mic-knl intel/18.0.3 impi/18.0.3 python/3.7

cd ${PBS_O_WORKDIR}


#ln -s /home01/r844a02/solver_BOSS ./solver_BOSS
#ln -s /home01/r844a02/obabel ./obabel

#export LD_LIBRARY_PATH=$HOME/obabel/lib:$LD_LIBRARY_PATH

mpirun -n 1 ./test
mpirun -n 4 ./test
mpirun -n 16 ./test
mpirun -n 32 ./test

