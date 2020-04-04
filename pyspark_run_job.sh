#!/bin/bash
#SBATCH --account=def-glatard
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
module load spark/2.4.4
module load python/3.7
# Recommended settings for calling Intel MKL routines from multi-threaded applications
# https://software.intel.com/en-us/articles/recommended-settings-for-calling-intel-mkl-routines-from-multi-threaded-applications
export MKL_NUM_THREADS=1
export SPARK_IDENT_STRING=$SLURM_JOBID
export SPARK_WORKER_DIR=$SLURM_TMPDIR
export SLURM_SPARK_MEM_FLOAT=$(echo "${SLURM_MEM_PER_NODE} * 0.95" | bc)
export SLURM_SPARK_MEM=${SLURM_SPARK_MEM_FLOAT%.*}
start-master.sh
sleep 5
MASTER_URL=$(grep -Po '(?=spark://).*' $SPARK_LOG_DIR/spark-${SPARK_IDENT_STRING}-org.apache.spark.deploy.master*.out)
NWORKERS=$((SLURM_NTASKS - 1))
SPARK_NO_DAEMONIZE=1 srun -n ${NWORKERS} -N ${NWORKERS} --label --output=$SPARK_LOG_DIR/spark-%j-workers.out start-slave.sh -m ${SLURM_SPARK_MEM}M -c ${SLURM_CPUS_PER_TASK}
${MASTER_URL} &
slaves_pid=$!

srun -n 1 -N 1 spark-submit --master ${MASTER_URL} --executor-memory ${SLURM_SPARK_MEM}M /home/jonmong/projects/def-glatard/jonmong/big-data/runners/random_forest.py

kill $slaves_pid
stop-master.sh
