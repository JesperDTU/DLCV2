"""
Utility functions for training and evaluation.
"""

import os
import shutil
import torch


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output: Model predictions [batch_size, num_classes]
        target: Ground truth labels [batch_size]
        topk: Tuple of k values

    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, checkpoint_path, checkpoint_dir, model_name):
    """
    Save model checkpoint.

    Args:
        state: Dictionary containing model state, optimizer state, etc.
        is_best: Boolean indicating if this is the best model so far
        checkpoint_path: Path to save the checkpoint
        checkpoint_dir: Directory to save checkpoints
        model_name: Name of the model
    """
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
        shutil.copyfile(checkpoint_path, best_path)


def load_checkpoint(checkpoint_path):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary containing checkpoint data
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EarlyStopping:
    """
    Early stopping to stop training when validation performance stops improving.
    """

    def __init__(self, patience=10, min_delta=0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics that should increase (accuracy),
                  'min' for metrics that should decrease (loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


def freeze_layers(model, freeze_until=None):
    """
    Freeze model layers up to a certain point.

    Args:
        model: PyTorch model
        freeze_until: Layer name to freeze until (inclusive)
    """
    freeze = True
    for name, param in model.named_parameters():
        if freeze:
            param.requires_grad = False
        if freeze_until is not None and freeze_until in name:
            freeze = False


def unfreeze_all(model):
    """
    Unfreeze all model parameters.

    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    # Test utilities
    import torch.nn as nn

    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")

    # Test accuracy
    output = torch.randn(4, 10)
    target = torch.randint(0, 10, (4,))
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    print(f"Top-1 Accuracy: {acc1.item():.2f}%")
    print(f"Top-5 Accuracy: {acc5.item():.2f}%")

    # Test parameter counting
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    print(f"Model parameters: {count_parameters(model)}")
