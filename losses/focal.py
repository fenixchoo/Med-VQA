import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, class_freq, gamma=1.0):
        super().__init__()
        w = class_freq.sum() / (len(class_freq) * class_freq)
        self.alpha = (w / w.mean()).to(class_freq.device)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze()
        return (self.alpha[targets] * (1 - pt) ** self.gamma * ce).mean()
