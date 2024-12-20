import torch
from pytorch_wavelets import DWT2D
import torch.nn as nn

class DWTLayer(nn.Module):
    def __init__(self, wavelet_):
        super(DWTLayer, self).__init__()
        self.dwt = DWT2D(J=1, wave=wavelet_)

    def forward(self, x):
        num_channels = x.shape[1]
        ll_lis, lh_lis, hl_lis = [], [], []
        for i in range(num_channels):
            channel = x[:, i, :, :].unsqueeze(1)
            low, high = self.dwt(channel)
            lh, hl, hh = torch.unbind(high[0], dim=2)
            ll_lis.append(low)
            lh_lis.append(lh)
            hl_lis.append(hl)

        ll = torch.stack(ll_lis, dim=1).squeeze(dim=2)
        hl = torch.stack(hl_lis, dim=1).squeeze(dim=2)
        lh = torch.stack(lh_lis, dim=1).squeeze(dim=2)

        return ll, lh, hl


class WaveletAttentionBlock(nn.Module):
    def __init__(self, wavelet='haar'):
        super(WaveletAttentionBlock, self).__init__()
        self.dwt = DWTLayer(wavelet)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        ll, lh, hl = self.dwt(x)
        out = (ll * self.softmax(lh + hl)) + ll
        return out

class WaveletAttentionStride(nn.Module):
    def __init__(self, in_features, out_features, wavelet='haar'):
        super(WaveletAttentionStride, self).__init__()
        self.att_block = WaveletAttentionBlock(wavelet)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.att_block(x)
        return self.conv(out)


if __name__ == '__main__':
    x = torch.randn(32, 3, 32, 32)
    blk = WaveletAttentionStride(3, 64, wavelet='haar')
    print(blk(x).shape)
