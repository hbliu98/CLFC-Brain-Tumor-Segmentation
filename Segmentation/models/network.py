import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.blocks(x)


class CLFC(nn.Module):
    def __init__(self, nf) -> None:
        super().__init__()
        self.zeta = nn.Conv2d(nf, nf//2, 1, 1, 0)
        self.g = nn.Conv2d(nf, nf//2, 1, 1, 0)
        self.phi = nn.Conv2d(nf//2, 1, 1, 1, 0)
    def forward(self, ft, fr):
        z_ft = self.zeta(ft)
        g_fr = self.g(fr)
        a = torch.sigmoid(self.phi(torch.relu(-z_ft*g_fr)))
        if self.training:
            return ft*a, (z_ft, g_fr)
        else:
            return ft*a


class projection(nn.Module):
    def __init__(self, nf) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(nf, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )
    def forward(self, x):
        x = x.permute(1, 0, 2, 3).flatten(1).T
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class prediction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(128, 256)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class NormNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.ModuleList()
        self.encoder.extend([
            ConvBlock(1, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512)
        ])
        self.decoder = nn.ModuleList()
        self.decoder.extend([
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.ConvTranspose2d(768, 256, 4, 2, 1),
            nn.ConvTranspose2d(384, 128, 4, 2, 1),
            nn.ConvTranspose2d(128, 64, 4, 2, 1)
        ])
        self.downsample = nn.MaxPool2d(2)
    def forward(self, x):
        e0 = self.downsample(self.encoder[0](x))
        e1 = self.downsample(self.encoder[1](e0))
        e2 = self.downsample(self.encoder[2](e1))
        e3 = self.downsample(self.encoder[3](e2))

        d0 = self.decoder[0](e3)
        d1 = self.decoder[1](torch.cat([d0, e2], dim=1))
        d2 = self.decoder[2](torch.cat([d1, e1], dim=1))
        d3 = self.decoder[3](d2)

        return [d0, d1, d2, d3]


class SegNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.ModuleList()
        self.encoder.extend([
            ConvBlock(4, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512)
        ])
        self.decoder = nn.ModuleList()
        self.decoder.extend([
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.ConvTranspose2d(768, 256, 4, 2, 1),
            nn.ConvTranspose2d(384, 128, 4, 2, 1),
            nn.ConvTranspose2d(128, 64, 4, 2, 1)
        ])
        self.conv_seg = nn.Conv2d(64, 2, 3, 1, 1)
        self.clfc = nn.ModuleList()
        self.clfc.extend([
            CLFC(512),
            CLFC(256),
            CLFC(128),
            CLFC(64)
        ])
        self.normnet = NormNet()
        self.downsample = nn.MaxPool2d(2)
    def forward(self, x, rec):
        fr = self.normnet(rec)

        e0 = self.downsample(self.encoder[0](x))
        e1 = self.downsample(self.encoder[1](e0))
        e2 = self.downsample(self.encoder[2](e1))
        e3 = self.downsample(self.encoder[3](e2))

        if not self.training:
            d0 = self.decoder[0](e3)
            d0 = self.clfc[0](d0, fr[0])
            d1 = self.decoder[1](torch.cat([d0, e2], dim=1))
            d1 = self.clfc[1](d1, fr[1])
            d2 = self.decoder[2](torch.cat([d1, e1], dim=1))
            d2 = self.clfc[2](d2, fr[2])
            d3 = self.decoder[3](d2)
            d3 = self.clfc[3](d3, fr[3])
            return self.conv_seg(d3)
        else:
            embeds = []
            d0 = self.decoder[0](e3)
            d0, embed = self.clfc[0](d0, fr[0])
            embeds.append(embed)
            d1 = self.decoder[1](torch.cat([d0, e2], dim=1))
            d1, embed = self.clfc[1](d1, fr[1])
            embeds.append(embed)
            d2 = self.decoder[2](torch.cat([d1, e1], dim=1))
            d2, embed = self.clfc[2](d2, fr[2])
            embeds.append(embed)
            d3 = self.decoder[3](d2)
            d3, embed = self.clfc[3](d3, fr[3])
            embeds.append(embed)
            embeds.reverse()
            return self.conv_seg(d3), embeds
