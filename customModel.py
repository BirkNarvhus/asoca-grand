import torch
import torch.nn as nn


class Resunit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.project = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.resunit = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
        )
        self.project_down = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x):
        res = self.project(x)
        return self.project_down(torch.cat([self.resunit(x), res], dim=1))


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
    def __init__(self, in_channels, out_channels, channels, strides=(2, 2, 2), *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.downlayers = nn.ModuleList()

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
            downblock.append(Resunit(_outChannel, _outChannel))
            if strides[i] != 1:
                downblock.append(nn.AvgPool3d(kernel_size=strides[i]))
            self.downlayers.append(nn.Sequential(*downblock))

        self.bottle = Bottleneck(channels[-1])
        self.uplayers = nn.ModuleList()

        for i in range(len(channels)):
            _inChannel = channels[-(i + 1)]

            _outChannel = channels[-(i + 2)] if i != len(channels) - 1 else out_channels

            upblock = nn.ModuleList()
            upblock.append(nn.Conv3d(_inChannel*2, _inChannel, kernel_size=1))
            upblock.append(Resunit(_inChannel, _outChannel))
            if strides[-(i+1)] != 1:
                upblock.append(nn.Upsample(scale_factor=strides[-(i+1)]))
            upblock.append(Resunit(_outChannel, _outChannel))
            upblock.append(Resunit(_outChannel, _outChannel))
            self.uplayers.append(nn.Sequential(*upblock))

    def forward(self, x):
        buffer = []
        for layer in self.downlayers:
            x = layer(x)
            buffer.append(x.detach().cpu())

        x = self.bottle(x)
        for layer in self.uplayers:
            buffer_x = (buffer.pop()).to(x.device)
            x = torch.cat([x, buffer_x], dim=1)
            x = layer(x)

        return x

def test():
    model = CustomModel(1, 1, [16, 32, 64], strides=(2, 2, 2))
    x = torch.randn(1, 1, 256, 256, 112)
    y = model(x)
    assert y.shape == torch.Size([1, 1, 256, 256, 112])


if __name__ == "__main__":
    test()