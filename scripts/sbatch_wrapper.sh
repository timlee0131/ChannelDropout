#!/usr/bin/env zsh

# Usage: DATA=Traffic ./sbatch_wrapper.sh

TIME=${TIME:-0-04:00:00}
PARTITION=${PARTITION:-batch}
GPU=${GPU:-1}
MEM=${MEM:-"40G"}
DATA=${DATA:-"Traffic"}

DATETIME=$(date +%Y_%m_%d_%H_%M_%S)
JOB_TYPE="ChannelDropout_${DATA}"

# write sbatch script
echo "#!/usr/bin/env zsh
#SBATCH -A coreyc_coreyc_mp_jepa_0001
#SBATCH -o ./logs/${DATA}_%j.out
#SBATCH -c 16 --mem=${MEM}
#SBATCH --nodes=1
#SBATCH -G ${GPU}
#SBATCH --time=${TIME} 
#SBATCH --partition=${PARTITION}

module purge
module load conda
conda activate /users/hunjael/.conda/envs/deltaformer

which python
echo \$CONDA_PREFIX

cd /users/hunjael/Projects/mts_codebases/ChannelDropout
srun bash ./scripts/long_term_forecasting/${DATA}/ChannelDropout.sh
" > ${JOB_TYPE}_${DATETIME}.sbatch

# submit sbatch script
sbatch ${JOB_TYPE}_${DATETIME}.sbatch

sleep 0.1
rm -f ${JOB_TYPE}_${DATETIME}.sbatch 