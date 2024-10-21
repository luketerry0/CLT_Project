#!/usr/bin/env bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=CLT_test
#SBATCH --output=/ourdisk/hpc/ai2es/luketerry/clt/computational_learning_theory_project/%j_0_log.out
#SBATCH --error=/ourdisk/hpc/ai2es/luketerry/clt/computational_learning_theory_project/%j_0_log.err
#SBATCH --time=24:00:00
#SBATCH --signal=USR2@300
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=10

EXPDIR=/ourdisk/hpc/ai2es/luketerry/clt/computational_learning_theory_project/
cd /ourdisk/hpc/ai2es/luketerry/clt/computational_learning_theory_project/

# # using Dr. Fagg's conda setup script
. /home/fagg/tf_setup.sh
# activating a version of my environment
conda activate /home/jroth/.conda/envs/mct

python test.py