"""
Model architectures for fault segmentation.
Uses segmentation_models_pytorch (SMP) for UNet, UNet++, DeepLabV3+.
Also SegFormer via SMP.

Verified reference solution uses smp.Unet.
We extend to ensemble of architectures for leaderboard top.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def get_model(arch="unetplusplus", encoder="efficientnet-b5", in_channels=10, classes=1, pretrained=True):
    arch = arch.lower()
    if arch == "unet":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif arch == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif arch == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif arch == "segformer":
        # SegFormer uses mit encoders
        # Map encoder name
        if "mit" not in encoder:
            encoder = "mit_b2"
        model = smp.Segformer(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif arch == "fpn":
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    else:
        raise ValueError(f"Unknown arch {arch}")

    return model

class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0/len(models)] * len(models)
        self.weights = weights

    def forward(self, x):
        # Average logits
        logits = 0
        for model, w in zip(self.models, self.weights):
            logits = logits + model(x) * w
        return logits

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
