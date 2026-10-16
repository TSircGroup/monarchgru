#!/bin/bash
#SBATCH --job-name=MN4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g
#SBATCH --gpus=1
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm-mn4-1-%j.out

module --force purge
module load mamba
source activate monarch-gru-env
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# For the first job only, add --download (it fetches the dataset into ./data; see README, Data). Remove it
# afterwards: it deletes and re-downloads the dataset folders (main.py refuses it when ./data is a link).
# The download runs in this job, so the compute node needs internet access. Submit other jobs only after
# this one has printed "Download and 12-class split finished." (in its slurm-mn4-1-<job id>.out).


python -u main.py \
  --hidden_size 64 \
  --nblocks 4 \
  --num_layers 2 \
  --layer_dropout 0.0 \
  --epochs 1000 \
  --batch_size 128 \
  --lr 0.005 \
  --weight_decay 0.03 \
  --grad_clip 4.0 \
  --warmup_epochs 20 \
  --label_smoothing 0.05 \
  --lr_schedule cos \
  --optimizer adamw \
  --n_mels 16 \
  --use_deltas \
  --specaug \
  --gpu 0 \
  --seed 456 \
  --freq_mask 7 \
  --time_mask 20 \
  --use_swa \
  --swa_start_frac 0.98 \
  --valid_silence_noise \
  --resample_unknown \
  --norm dual \
  --dense_input_layer1 \
  --deterministic \
  --run_name $SLURM_JOB_ID
