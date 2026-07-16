#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A Catalyst
#PBS -l filesystems=home:grand:eagle

#module swap PrgEnv-nvhpc PrgEnv-gnu
#module load cudatoolkit-standalone

cd /lus/grand/projects/catalyst/proj-shared/knight/projects/Voth/lammps/examples/PACKAGES/raptor/single

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
NDEPTH=1
NTHREADS=1
NGPUS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

EXE=/lus/grand/projects/catalyst/proj-shared/knight/projects/Voth/lammps/src/lmp_polaris_gnu

LMP_ARGS="-in lmp.in"
LMP_ARGS="-in lmp_gpu.in -pk gpu ${NGPUS}"

COMMAND="mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth /home/knight/scripts/polaris/set_affinity_gpu_polaris.sh ${EXE} ${LMP_ARGS}"
#COMMAND="mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth /home/knight/scripts/polaris/set_affinity_gpu_polaris.sh ncu ${EXE} ${LMP_ARGS}"
#COMMAND="mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth /home/knight/scripts/polaris/set_affinity_gpu_polaris.sh nsys profile ${EXE} ${LMP_ARGS}"
echo "COMMAND= ${COMMAND}"
${COMMAND}
