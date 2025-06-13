#!/usr/bin/env zsh

module purge
module load conda
conda activate /users/hunjael/.conda/envs/deltaformer

srun\
    -N1\
    -G1\
    -p short\
    -A coreyc_coreyc_mp_jepa_0001\
    --pty $SHELL\