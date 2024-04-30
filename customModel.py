import torch
import torch.nn as nn


class Resunit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.project = nn.Identity()
        hidden = out_channels//2
        if in_channels != hidden:
            self.project = nn.Conv3d(in_channels, hidden, kernel_size=1)
        self.resunit = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(hidden),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(hidden),
        )

    def forward(self, x):
        x = self.project(x)
        return torch.cat([x, self.resunit(x)], dim=1)


class Bottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, dilation=1),
            nn.Conv3d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.Conv3d(channels, channels, kernel_size=3, padding=4, dilation=4),
        )

    def forward(self, x):
        buffer = x
        for layer in self.bottleneck:
            x = layer(x)
            buffer = buffer + x
        return buffer


class CustomModel(nn.Module):
    def __init__(self, in_channels, out_channels, channels, strides=(2, 2, 2), device='cpu'):

        super().__init__()
        self.downlayers = nn.ModuleList()
        self.device = device
        self.to(device)

        if len(channels) != len(strides):
            raise ValueError("channels and strides must have the same length")

        for i in range(len(channels)):
            if i == 0:
                _inChannel = in_channels
            else:
                _inChannel = channels[i - 1]
            _outChannel = channels[i]
            downblock = nn.ModuleList()
            downblock.append(Resunit(_inChannel, _outChannel))
            if strides[i] != 1:
                downblock.append(nn.AvgPool3d(kernel_size=strides[i]))
            self.downlayers.append(nn.Sequential(*downblock))

        self.bottle = Bottleneck(channels[-1])
        self.uplayers = nn.ModuleList()

        for i in range(len(channels)):
            _inChannel = channels[-(i + 1)]

            _outChannel = channels[-(i + 2)] if i != len(channels) - 1 else out_channels

            upblock = nn.ModuleList()
            if strides[-(i+1)] != 1:
                upblock.append(nn.Upsample(scale_factor=strides[-(i+1)]))

            upblock.append(nn.Conv3d(_inChannel*2, _outChannel, kernel_size=1))
            upblock.append(nn.LeakyReLU())
            upblock.append(nn.BatchNorm3d(_outChannel))

            upblock.append(nn.Conv3d(_outChannel, _outChannel, kernel_size=3, padding=1))
            upblock.append(nn.LeakyReLU())
            upblock.append(nn.BatchNorm3d(_outChannel))

            upblock.append(nn.Conv3d(_outChannel, _outChannel, kernel_size=3, padding=1))
            upblock.append(nn.LeakyReLU())
            upblock.append(nn.BatchNorm3d(_outChannel))

            self.uplayers.append(nn.Sequential(*upblock))
        self.uplayers = nn.Sequential(*self.uplayers)
    def forward(self, x):
        buffer = []
        for layer in self.downlayers:
            x = layer(x)
            buffer.append(x.detach().cpu())

        #x = self.bottle(x)
        for a, layer in zip(buffer[::-1], self.uplayers):
            x = layer(torch.cat([x, a.to(self.device)], dim=1))

        return x

def test():
    model = CustomModel(1, 1, [16, 32, 64], strides=(4, 2, 2))




if __name__ == "__main__":
    test()