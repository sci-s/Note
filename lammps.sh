#!/bin/bash
#PBS -N ii
#PBS -q samdi2
#PBS -A etc
#PBS -l select=1:ncpus=32:mpiprocs=32:ompthreads=1
#PBS -l walltime=00:30:00

module purge
module load intel/19.0.5 impi/19.0.5 craype-mic-knl lammps/3Mar20

cd ${PBS_O_WORKDIR}


#ln -s /home01/r844a02/solver_BOSS ./solver_BOSS
#ln -s /home01/r844a02/obabel ./obabel

#export LD_LIBRARY_PATH=$HOME/obabel/lib:$LD_LIBRARY_PATH

mpirun -n 32 lmp_mpi -in in.lj
