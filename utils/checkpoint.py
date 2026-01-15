import os
import torch

def save_checkpoint(
    model,
    optimizer,
    epoch,
    path,
    scheduler=None,
    best_metric=None,
):
    """
    Save training checkpoint.

    Args:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        epoch: int
        path: str (file path, not directory)
        scheduler: optional LR scheduler
        best_metric: optional float (e.g. best val accuracy)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    if best_metric is not None:
        ckpt["best_metric"] = best_metric

    torch.save(ckpt, path)


def load_checkpoint(
    model,
    optimizer,
    path,
    scheduler=None,
    device="cpu",
):
    """
    Load checkpoint and restore states.

    Returns:
        start_epoch (int)
        best_metric (float or None)
    """
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt.get("epoch", -1) + 1
    best_metric = ckpt.get("best_metric", None)

    return start_epoch, best_metric
