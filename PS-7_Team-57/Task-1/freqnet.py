import torch
import torch.nn as nn


def HFRIBlock(x):
    """
    High Frequency Representation of Image
    """
    freq_represnetation = torch.fft.fft2(x, dim=(-2, -1))
    freq_represnetation[..., :x.shape[-2] // 4, :] = 0
    freq_represnetation[..., :, :x.shape[-1] // 4] = 0
    processed_image = torch.fft.ifft2(freq_represnetation, dim=(-2, -1)).real
    return processed_image


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.conv(x)


class HFRFSBlock(nn.Module):
    """
    High Frequency Representation of features across spatial
    """
    def __init__(self, in_channels, out_channels):
        super(HFRFSBlock, self).__init__()
        self.conv = ConvLayer(in_channels, out_channels)

    def forward(self, x):
        spatial_freq = torch.fft.fft2(x, dim=(-2, -1))
        spatial_freq[..., :x.shape[-2] // 4, :] = 0
        spatial_freq[..., :, :x.shape[-1] // 4] = 0
        spatial_processed = torch.fft.ifft2(spatial_freq, dim=(-2, -1)).real

        return self.conv(spatial_processed)


class HFRFCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HFRFCBlock, self).__init__()
        self.conv = ConvLayer(in_channels, out_channels)

    def forward(self, x):
        channel_freq = torch.fft.fft(x, dim=1)
        channel_freq[:, :x.shape[1] // 4, :, :] = 0
        channel_processed = torch.fft.ifft(channel_freq, dim=1).real

        return self.conv(channel_processed)


class FCLBlock(nn.Module):
    """
    Frequency Convolution Layer
    """
    def __init__(self, in_channels, out_channels):
        super(FCLBlock, self).__init__()
        self.residual_conv_start = ConvLayer(in_channels, out_channels)
        self.phase_conv = ConvLayer(out_channels, out_channels)
        self.amplitude_conv = ConvLayer(out_channels, out_channels)
        self.residual_conv_final = ConvLayer(out_channels, out_channels)

    def forward(self, x):
        x = self.residual_conv_start(x)
        freq_represnetation = torch.fft.fft2(x, dim=(-2, -1))
        amplitude = freq_represnetation.abs()
        phase = torch.angle(freq_represnetation)

        amp_processed = self.amplitude_conv(amplitude)
        phase_processed = self.phase_conv(phase)

        combined_freq = torch.polar(amp_processed, phase_processed)

        processed = torch.fft.ifft2(combined_freq, dim=(-2, -1)).real

        return self.residual_conv_final(processed)

class FreqNet(nn.Module):
    def __init__(self):
        super(FreqNet, self).__init__()
        # self.block1 = nn.Sequential(
        #     HFRFCBlock(3, 3),
        #     FCLBlock(3, 32)
        # )
        # self.block2 = nn.Sequential(
        #     HFRFSBlock(32, 32),
        #     HFRFCBlock(32, 32),
        #     FCLBlock(32, 64),
        #     ConvLayer(64, 64)
        # )
        self.block3 = nn.Sequential(
            HFRFSBlock(3, 3),
            FCLBlock(3, 64),
            HFRFSBlock(64, 64),
            FCLBlock(64, 128),
            ConvLayer(128, 128, stride=2)
        )
        self.final_conv = ConvLayer(128, 256, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, x):
        hfri_out = HFRIBlock(x)
        # out = self.block3(self.block2(self.block1(hfri_out)))
        out = self.final_conv(self.block3(hfri_out))
        # pool = self.avgpool(out)
        # return self.fc(pool.reshape(x.shape[0], -1))
        return out

def test_freqnet():
    model = FreqNet()
    print(model(torch.randn(1, 3, 32, 32)).shape)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    print(model)

if __name__ == '__main__':
    test_freqnet()








