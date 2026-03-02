#!/bin/bash
#SBATCH --job-name=ur5_act_train
#SBATCH --account=stf          # Account from hyakalloc
#SBATCH --partition=ckpt       # Using 'ckpt' to access the idle GPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1               # Requesting 1 GPU
#SBATCH --time=08:00:00        # Max time
#SBATCH --output=slurm_ur5_act_%j.out

# Load your environment
source /gscratch/scrubbed/architj/arc_hitter/miniconda3/etc/profile.d/conda.sh

# Activating project conda environment
conda activate ur5_act

export TORCH_HOME=/gscratch/scrubbed/architj/arc_hitter/.cache/torch
mkdir -p $TORCH_HOME


# 3. Print the node name (crucial for TensorBoard later)
echo "Job running on node: $(hostname)"

# Run your code
#python /gscratch/scrubbed/architj/arc_hitter/your_script.py
python -u /gscratch/scrubbed/architj/arc_hitter/ur5_act/training/train.py --config /gscratch/scrubbed/architj/arc_hitter/ur5_act/config/task_config.yaml --extracted_dir /gscratch/scrubbed/architj/arc_hitter/ur5_act/data/extracted

