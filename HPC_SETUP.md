# DTU HPC Training Setup Guide

This guide will help you set up and run your video classification training on the DTU HPC cluster.

## Prerequisites

- Access to DTU HPC cluster
- Project files uploaded to HPC
- Dataset available at `/dtu/datasets1/02516/ucf101` (or `/dtu/datasets1/02516/ucf101noleakage` for Project 4.2)

## Initial Setup

### 1. Connect to DTU HPC

```bash
ssh YOUR_DTU_ID@login.hpc.dtu.dk
```

### 2. Upload Your Project

From your local machine:

```bash
# Create a tar archive of your project
tar -czf DLCV2.tar.gz DLCV2/

# Upload to HPC
scp DLCV2.tar.gz YOUR_DTU_ID@transfer.hpc.dtu.dk:~/

# On HPC, extract the archive
ssh YOUR_DTU_ID@login.hpc.dtu.dk
tar -xzf DLCV2.tar.gz
cd DLCV2
```

Or use `rsync` for faster transfer:

```bash
rsync -avz --progress DLCV2/ YOUR_DTU_ID@transfer.hpc.dtu.dk:~/DLCV2/
```

### 3. Set Up Python Environment (First Time Only)

```bash
# Load Python module
module load python3/3.11.7
module load cuda/12.1

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create Logs Directory

```bash
mkdir -p logs checkpoints
```

## Submitting Training Jobs

### Quick Start - Submit All Jobs

To submit all 4 models for Project 4.1:

```bash
bash submit_all.sh
```

### Submit Individual Models

**Per-Frame CNN:**
```bash
bsub < submit_perframe.sh
```

**Late Fusion CNN:**
```bash
bsub < submit_latefusion.sh
```

**Early Fusion CNN:**
```bash
bsub < submit_earlyfusion.sh
```

**3D CNN:**
```bash
bsub < submit_3dcnn.sh
```

## Monitoring Jobs

### Check Job Status

```bash
# List your jobs
bstat

# Detailed status
bstat -u YOUR_DTU_ID

# Check specific job
bstat JOB_ID
```

### View Job Output

```bash
# View output in real-time
tail -f logs/perframe_JOBID.out

# View error log
tail -f logs/perframe_JOBID.err
```

### Check GPU Usage

```bash
# See GPU utilization for your job
nvidia-smi
```

## Managing Jobs

### Cancel a Job

```bash
bkill JOB_ID
```

### Cancel All Your Jobs

```bash
bkill 0
```

## Job Configuration Explained

The job scripts use LSF (Load Sharing Facility) directives:

- `#BSUB -q gpua100` - Use A100 GPU queue
- `#BSUB -J perframe_train` - Job name
- `#BSUB -n 4` - Number of CPU cores
- `#BSUB -gpu "num=1:mode=exclusive_process"` - Request 1 GPU
- `#BSUB -W 24:00` - Wall time limit (24 hours)
- `#BSUB -R "rusage[mem=16GB]"` - Memory per core
- `#BSUB -R "span[hosts=1]"` - Run on single host
- `#BSUB -o logs/perframe_%J.out` - Output file
- `#BSUB -e logs/perframe_%J.err` - Error file

## Adjusting Resources

If you encounter issues, you can modify the job scripts:

### Out of Memory

Increase memory in the job script:
```bash
#BSUB -R "rusage[mem=32GB]"
```

Or reduce batch size in training:
```bash
python3 train.py --model perframe --batch_size 16  # instead of 32
```

### Job Takes Too Long

Increase wall time:
```bash
#BSUB -W 48:00  # 48 hours instead of 24
```

Or reduce epochs:
```bash
python3 train.py --model perframe --epochs 30  # instead of 50
```

### Need Different GPU

Available GPU queues:
- `gpua100` - A100 GPUs (recommended)
- `gpuv100` - V100 GPUs
- `gpua40` - A40 GPUs

Change in job script:
```bash
#BSUB -q gpuv100
```

## Downloading Results

### From HPC to Local Machine

```bash
# Download checkpoints
scp -r YOUR_DTU_ID@transfer.hpc.dtu.dk:~/DLCV2/checkpoints ./

# Download logs
scp -r YOUR_DTU_ID@transfer.hpc.dtu.dk:~/DLCV2/logs ./

# Download specific model
scp YOUR_DTU_ID@transfer.hpc.dtu.dk:~/DLCV2/checkpoints/perframe_best.pth ./
```

## Training Tips for HPC

1. **Use Job Arrays** for hyperparameter search:
   ```bash
   #BSUB -J "train[1-5]"
   ```

2. **Check Dataset Path** - Verify the dataset exists:
   ```bash
   ls /dtu/datasets1/02516/ucf101/
   ```

3. **Test Locally First** - Run a quick test before submitting:
   ```bash
   # Interactive session
   bsub -Is -q gpua100 -n 4 -gpu "num=1" -W 1:00 -R "rusage[mem=8GB]" bash

   # Then test your code
   source venv/bin/activate
   python3 train.py --model perframe --epochs 1 --batch_size 8
   ```

4. **Monitor Progress** - Check logs regularly:
   ```bash
   watch -n 10 tail -20 logs/perframe_*.out
   ```

5. **Save Checkpoints** - Models automatically save to `checkpoints/`

## Project 4.2 - Using ucf101noleakage

For Task 4.2.1, use the dataset without information leakage:

```bash
# Update root_dir in job scripts to:
--root_dir /dtu/datasets1/02516/ucf101noleakage
```

Or create a new submission script:

```bash
cp submit_perframe.sh submit_perframe_noleakage.sh
# Edit the file and change root_dir
bsub < submit_perframe_noleakage.sh
```

## Troubleshooting

### Job Pending for Long Time

Check queue status:
```bash
bqueues
```

Consider using different queue if gpua100 is busy:
```bash
#BSUB -q gpuv100
```

### Import Errors

Make sure virtual environment is activated in job script:
```bash
source venv/bin/activate
```

### CUDA Out of Memory

Reduce batch size or use gradient accumulation:
```bash
python3 train.py --model 3dcnn --batch_size 4  # reduced from 8
```

### Dataset Not Found

Check if dataset path is correct:
```bash
ls -la /dtu/datasets1/02516/ucf101/
```

If using different dataset location, update `--root_dir` in job scripts.

## Quick Reference Commands

```bash
# Submit job
bsub < submit_perframe.sh

# Check status
bstat

# Cancel job
bkill JOB_ID

# View output
tail -f logs/perframe_*.out

# Interactive GPU session
bsub -Is -q gpua100 -n 4 -gpu "num=1" -W 2:00 -R "rusage[mem=16GB]" bash

# Download results
scp -r YOUR_DTU_ID@transfer.hpc.dtu.dk:~/DLCV2/checkpoints ./
```

## Additional Resources

- [DTU HPC Wiki](https://www.hpc.dtu.dk/)
- [LSF Documentation](https://www.ibm.com/docs/en/spectrum-lsf)
- Course dataset: `/dtu/datasets1/02516/`

## Support

If you encounter issues:
1. Check job logs in `logs/` directory
2. Review DTU HPC documentation
3. Ask TAs during office hours
4. Email HPC support: support@hpc.dtu.dk
