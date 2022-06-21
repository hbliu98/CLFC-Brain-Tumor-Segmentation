import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, ignore_bg=False) -> None:
        super().__init__()
        self.nonline = nn.Softmax(dim=1)
        self.smooth = 1e-5
        self.ignore_bg = ignore_bg
    
    def forward(self, x, y):
        with torch.no_grad():
            y_onehot = torch.zeros_like(x)
            y_onehot = y_onehot.scatter(1, y.long(), 1)
        axes = [0] + list(range(2, len(x.shape)))
        
        x = self.nonline(x)

        tp = (x * y_onehot).sum(axes)
        fp = (x * (1 - y_onehot)).sum(axes)
        fn = ((1 - x) * y_onehot).sum(axes)
        
        numerator = 2. * tp + self.smooth
        denominator = 2. * tp + fp + fn + self.smooth
        dc = numerator / (denominator + 1e-8)
        dc = dc[1:].mean() if self.ignore_bg else dc.mean()
        return -dc


class DiceCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dc = DiceLoss(ignore_bg=True)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, y):
        dc_loss = self.dc(x, y)
        ce_loss = self.ce(x, y.squeeze(1).long())
        return dc_loss + ce_loss
