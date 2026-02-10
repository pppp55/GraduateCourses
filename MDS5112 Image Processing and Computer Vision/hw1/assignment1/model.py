import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
    from torchvision.models import ResNet50_Weights
except Exception:  # pragma: no cover
    deeplabv3_resnet50 = None
    fcn_resnet50 = None
    ResNet50_Weights = None


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def build_segmentation_model(
    arch: str,
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained_backbone: bool = True,
) -> nn.Module:
    arch = arch.lower()
    if arch == "unet":
        return UNet(n_channels=in_channels, n_classes=num_classes)

    if fcn_resnet50 is None or deeplabv3_resnet50 is None:
        raise ImportError("torchvision is required for fcn/deeplab architectures.")

    if in_channels != 3:
        raise ValueError(
            "torchvision segmentation models currently assume 3-channel input."
        )

    if arch == "fcn_resnet50":
        weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        return fcn_resnet50(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes
        )

    if arch == "deeplabv3_resnet50":
        weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        return deeplabv3_resnet50(
            weights=None, weights_backbone=weights_backbone, num_classes=num_classes
        )

    raise ValueError(
        f"Unknown arch: {arch}. Supported: unet, fcn_resnet50, deeplabv3_resnet50"
    )
