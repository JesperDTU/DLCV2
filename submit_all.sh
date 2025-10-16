#!/bin/bash
# Submit all training jobs for Project 4.1

echo "Submitting all training jobs to DTU HPC..."

# Submit per-frame CNN
echo "Submitting Per-Frame CNN..."
bsub < submit_perframe.sh

# Submit late fusion
echo "Submitting Late Fusion CNN..."
bsub < submit_latefusion.sh

# Submit early fusion
echo "Submitting Early Fusion CNN..."
bsub < submit_earlyfusion.sh

# Submit 3D CNN
echo "Submitting 3D CNN..."
bsub < submit_3dcnn.sh

echo ""
echo "All jobs submitted! Check status with: bstat"
echo "View logs in: logs/"
