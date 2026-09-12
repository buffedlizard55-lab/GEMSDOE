"""
Losses matching metric: Tversky with alpha=0.2, beta=0.8 to penalize FN more.
Also Focal Tversky, Dice, BCE.
Verified sources: https://en.wikipedia.org/wiki/Tversky_index
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, smooth=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B,1,H,W) or (B,H,W)
        # targets: same shape, binary
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        probs = torch.sigmoid(logits)  # inputs are raw logits (explicit; range-sniffing removed)
        # (if you ever pass probabilities instead, wrap: loss(torch.logit(p.clamp(1e-6,1-1e-6)), t))
        # flatten
        probs = probs.view(probs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1).float()
        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1 - targets)).sum(dim=1)
        FN = ((1 - probs) * targets).sum(dim=1)
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky.mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, gamma=0.75, smooth=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        probs = torch.sigmoid(logits)  # inputs are raw logits (explicit; range-sniffing removed)
        # (if you ever pass probabilities instead, wrap: loss(torch.logit(p.clamp(1e-6,1-1e-6)), t))
        probs = probs.view(probs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1).float()
        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1 - targets)).sum(dim=1)
        FN = ((1 - probs) * targets).sum(dim=1)
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal = (1 - tversky) ** self.gamma
        return focal.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, bce_weight=0.5, tversky_weight=0.5, focal_gamma=0.75):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.focal_tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=focal_gamma)
        self.bce_w = bce_weight
        self.tversky_w = tversky_weight

    def forward(self, logits, targets):
        # BCE expects logits
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        if logits.shape != targets.shape:
            # logits (B,1,H,W) targets (B,H,W) -> unsqueeze
            if logits.dim() == 4 and targets.dim() == 3:
                targets = targets.unsqueeze(1)
        bce_loss = self.bce(logits, targets.float())
        # Tversky expects probs or logits; we pass logits
        t_loss = self.tversky(logits, targets)
        f_loss = self.focal_tversky(logits, targets)
        return self.bce_w * bce_loss + self.tversky_w * (0.5 * t_loss + 0.5 * f_loss)
