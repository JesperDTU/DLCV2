#!/bin/bash
#BSUB -q gpua100
#BSUB -J earlyfusion_train
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/earlyfusion_%J.out
#BSUB -e logs/earlyfusion_%J.err

# Load modules
module load python3/3.11.7
module load cuda/12.1

# Navigate to project directory
cd $HOME/DLCV2

# Activate virtual environment
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training
python3 train.py \
    --root_dir /dtu/datasets1/02516/ucf101 \
    --model earlyfusion \
    --backbone resnet18 \
    --batch_size 16 \
    --epochs 50 \
    --lr 0.001 \
    --num_workers 4

echo "Training completed!"
