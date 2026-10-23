"""ResNet34 U-Net and training loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DC(nn.Sequential):
    def __init__(s, i, o):
        super().__init__(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class Up(nn.Module):
    def __init__(s, i, sk, o):
        super().__init__()
        s.c = DC(i + sk, o)

    def forward(s, x, sk):
        return s.c(torch.cat([F.interpolate(x, size=sk.shape[2:], mode="bilinear",
                                            align_corners=False), sk], 1))


class Net(nn.Module):
    """ImageNet-pretrained ResNet34 encoder, U-Net decoder (256,128,64,64), 1x1 output head.

    out_ch = number of measurements (+1 auxiliary ellipse-mask channel for head/abdomen).
    Output resolution is 128x128 (stride 2).
    """

    def __init__(s, out_ch):
        super().__init__()
        try:
            e = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        except Exception as ex:
            print("no pretrained weights:", str(ex)[:60])
            e = models.resnet34(weights=None)
        s.stem = nn.Sequential(e.conv1, e.bn1, e.relu)
        s.pool = e.maxpool
        s.l1, s.l2, s.l3, s.l4 = e.layer1, e.layer2, e.layer3, e.layer4
        s.u4, s.u3, s.u2, s.u1 = Up(512, 256, 256), Up(256, 128, 128), Up(128, 64, 64), Up(64, 64, 64)
        s.out = nn.Conv2d(64, out_ch, 1)
        s.register_buffer("mu", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        s.register_buffer("sd", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(s, x):
        x = (x - s.mu) / s.sd
        x0 = s.stem(x)
        x1 = s.l1(s.pool(x0))
        x2 = s.l2(x1)
        x3 = s.l3(x2)
        x4 = s.l4(x3)
        u4 = s.u4(x4, x3)
        u3 = s.u3(u4, x2)
        u2 = s.u2(u3, x1)
        u1 = s.u1(u2, x0)
        return s.out(u1)


def loss_fn(o, h, K, aux):
    """Weighted MSE on heatmaps, (1 + 10 h) x 100, plus 2 x (BCE + soft Dice) on the ellipse mask."""
    l = (((o[:, :K] - h[:, :K]) ** 2) * (1 + 10 * h[:, :K])).mean() * 100
    if aux:
        p, t = o[:, K], h[:, K]
        sg = torch.sigmoid(p)
        dice = 1 - (2 * (sg * t).sum((1, 2)) + 1) / (sg.sum((1, 2)) + t.sum((1, 2)) + 1)
        l = l + 2.0 * (F.binary_cross_entropy_with_logits(p, t) + dice.mean())
    return l
