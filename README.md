# Video Classification Project - UCF-101

Deep learning project for video action recognition using a subset of the UCF-101 dataset.

## Project Overview

This project implements various video classification approaches for recognizing workout actions from videos:

**10 Action Classes:**
- BodyWeightSquats
- HandstandPushups
- HandstandWalking
- JumpingJack
- JumpRope
- Lunges
- PullUps
- PushUps
- TrampolineJumping
- WallPushups

**Dataset:** 720 videos (500/120/120 train/val/test), each with 10 uniformly sampled frames.

## Project Structure

```
DLCV2/
├── datasets.py         # Dataset classes for frames and videos
├── models.py           # Model architectures
├── train.py            # Training script
├── utils.py            # Utility functions
├── data/               # Dataset directory (create this)
│   ├── frames/         # Video frames
│   ├── flows-png/      # Optical flow (for dual-stream)
│   └── metadata/       # CSV files (train.csv, val.csv, test.csv)
├── checkpoints/        # Saved model checkpoints
├── logs/               # Training logs
└── models/             # Additional model implementations
```

## Requirements

Install dependencies:

```bash
pip install torch torchvision pandas pillow tqdm
```

## Models Implemented

### Project 4.1: Video Classification Approaches

1. **Per-Frame CNN**: Process each frame independently and aggregate predictions
   - Aggregation: mean/max pooling over predictions

2. **Late Fusion**: Extract features from each frame, combine before classification
   - Fusion: concatenate or average features

3. **Early Fusion**: Stack all frames in channel dimension, process with 2D CNN
   - Input: [B, C×T, H, W]

4. **3D CNN**: Use 3D convolutions to capture spatiotemporal features
   - Input: [B, C, T, H, W]

### Project 4.2: Advanced Topics

5. **Dual-Stream Network**: Two-stream architecture with RGB and optical flow
   - Spatial stream: RGB frames
   - Temporal stream: Optical flow
   - Fusion: late (average predictions) or feature concatenation

## Usage

### 1. Prepare Dataset

Download the dataset from DTU Learn and organize as follows:

```
data/
├── frames/
│   ├── train/
│   │   ├── BodyWeightSquats/
│   │   │   ├── v_BodyWeightSquats_g01_c01/
│   │   │   │   ├── frame_1.jpg
│   │   │   │   ├── frame_2.jpg
│   │   │   │   └── ...
│   │   └── ...
│   ├── val/
│   └── test/
├── videos/
│   ├── train/
│   ├── val/
│   └── test/
├── flows-png/          # For dual-stream (Task 4.2.2)
│   └── ...
└── metadata/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

**Note:** For Project 4.2, use the `ucf101noleakage` dataset from `/dtu/datasets1/02516/` to avoid information leakage.

### 2. Train Models

**Per-Frame CNN:**
```bash
python train.py --model perframe --backbone resnet18 --batch_size 32 --epochs 50
```

**Late Fusion:**
```bash
python train.py --model latefusion --backbone resnet18 --fusion mean --batch_size 16
```

**Early Fusion:**
```bash
python train.py --model earlyfusion --backbone resnet18 --batch_size 16
```

**3D CNN:**
```bash
python train.py --model 3dcnn --batch_size 8 --epochs 50
```

**Dual-Stream Network (Task 4.2.2):**
```bash
python train.py --model dualstream --backbone resnet18 --fusion late --batch_size 8
```

### 3. Resume Training

```bash
python train.py --model perframe --resume checkpoints/perframe_latest.pth
```

### 4. Evaluate Only

```bash
python train.py --model perframe --resume checkpoints/perframe_best.pth --eval_only
```

## Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--root_dir` | Root directory of dataset | `/work3/ppar/data/ucf101` |
| `--model` | Model type: perframe, latefusion, earlyfusion, 3dcnn, dualstream | `perframe` |
| `--backbone` | Backbone: resnet18, resnet50 | `resnet18` |
| `--pretrained` | Use pretrained weights | `True` |
| `--batch_size` | Batch size | `16` |
| `--epochs` | Number of epochs | `50` |
| `--lr` | Learning rate | `0.001` |
| `--img_size` | Input image size | `224` |
| `--num_frames` | Frames per video | `10` |
| `--aggregation` | Aggregation for per-frame: mean, max | `mean` |
| `--fusion` | Fusion method: mean, concat, late | `mean` |

## Tips and Best Practices

### For Project 4.1:

1. **Start simple**: Begin with per-frame CNN to establish a baseline
2. **Data augmentation**: Use random flips, color jitter for better generalization
3. **Pretrained weights**: Start with ImageNet pretrained models
4. **Learning rate scheduling**: Use ReduceLROnPlateau for adaptive learning
5. **Batch size**: Smaller batch sizes for video models (8-16) due to memory

### For Project 4.2:

**Task 4.2.1 (Information Leakage):**
- Retrain your best model from 4.1 on `ucf101noleakage` dataset
- Compare performance to identify leakage impact
- Document the performance drop in your report

**Task 4.2.2 (Dual-Stream):**
- Pre-computed optical flows are in `flows-png/` directory
- Use late fusion (averaging predictions) as baseline
- Experiment with feature concatenation for better results
- Spatial and temporal streams can be pretrained separately

## Example Workflow

```bash
# 1. Train per-frame baseline
python train.py --model perframe --backbone resnet18 --epochs 30

# 2. Try late fusion
python train.py --model latefusion --backbone resnet18 --epochs 30

# 3. Experiment with 3D CNN
python train.py --model 3dcnn --batch_size 8 --epochs 40

# 4. For Task 4.2.2: Train dual-stream
python train.py --model dualstream --root_dir /dtu/datasets1/02516/ucf101noleakage --epochs 40
```

## Report Guidelines

Submit a 3-page report (CVPR template) including:

1. **Introduction**: Brief problem description
2. **Methods**: Model architectures you implemented
3. **Experiments**:
   - Ablation studies comparing different approaches
   - Training curves (loss, accuracy)
   - Performance tables
4. **Results & Discussion**:
   - Best model performance
   - Impact of information leakage (Task 4.2.1)
   - Dual-stream improvements (Task 4.2.2)
5. **Conclusion**: Key findings and takeaways

**Note:** References and AI usage statement don't count toward 3-page limit.

## Troubleshooting

**Out of Memory:**
- Reduce batch size
- Use smaller backbone (resnet18 instead of resnet50)
- Reduce image size to 112×112

**Poor Performance:**
- Check data augmentation
- Verify pretrained weights are loaded
- Try different learning rates (1e-3, 1e-4)
- Ensure proper normalization

**Slow Training:**
- Increase num_workers for data loading
- Use GPU if available
- Enable pin_memory in DataLoader

## Additional Resources

- [UCF-101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
- [Two-Stream Networks Paper](https://arxiv.org/abs/1406.2199)
- [RAFT Optical Flow](https://github.com/princeton-vl/RAFT)

## License

This is an educational project for DTU course 02516: Introduction to Deep Learning in Computer Vision.
