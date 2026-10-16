#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# The job runs on the CPU. To use a GPU, remove the space in '# SBATCH' on the next two lines and set a GPU
# partition of your cluster.
# SBATCH --partition=gpu-v100-32g,gpu-v100-16g
# SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-inference-%j.out

module --force purge
module load mamba
source activate monarch-gru-env

# To classify your own clips instead, replace --eval_test with the .wav files (e.g. my_folder/*.wav).
python -u inference.py --eval_test
