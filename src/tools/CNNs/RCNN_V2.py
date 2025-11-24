import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- SpecAugment (same as before) ----------
def spec_augment(x, time_mask=32, freq_mask=16, p=1.0):
    if torch.rand(1).item() > p:
        return x
    B, _, Freq, T = x.shape

    f = torch.randint(0, freq_mask + 1, (1,)).item()
    if f > 0:
        f0 = torch.randint(0, max(1, Freq - f), (1,)).item()
        x[:, :, f0:f0 + f, :] = 0.0

    t = torch.randint(0, time_mask + 1, (1,)).item()
    if t > 0:
        t0 = torch.randint(0, max(1, T - t), (1,)).item()
        x[:, :, :, t0:t0 + t] = 0.0
    return x


# --------- Light music-aware residual block ----------
class AsymResBlock(nn.Module):
    """
    2-branch music-aware block:
      - branch A: (3x3) captures local textures
      - branch B: (1x7) captures rhythm/time patterns
    total channels kept small.
    """
    def __init__(self, Cin, Cout, p_drop=0.1):
        super().__init__()
        mid = Cout // 2

        self.b1 = nn.Sequential(
            nn.Conv2d(Cin, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(Cin, mid, kernel_size=(1,7), padding=(0,3), bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )

        self.mix = nn.Sequential(
            nn.Conv2d(2*mid, Cout, kernel_size=1, bias=False),
            nn.BatchNorm2d(Cout)
        )

        self.skip = None
        if Cin != Cout:
            self.skip = nn.Sequential(
                nn.Conv2d(Cin, Cout, kernel_size=1, bias=False),
                nn.BatchNorm2d(Cout)
            )

        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p_drop)

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b2(x)], dim=1)
        y = self.mix(y)
        s = x if self.skip is None else self.skip(x)
        return self.drop(self.act(y + s))


class CRNNv2(nn.Module):
    """
    Input:  x [B,1,128,T]  (T<=256 or 320 ok)
    Output: logits [B,n_classes]

    Memory-safe:
      - only 2 light branches per block
      - channels capped at 128 before projection
      - freq-aware collapse before GRU (no huge flatten)
    """
    def __init__(self, n_classes, rnn_hidden=128):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )

        self.block1 = AsymResBlock(32, 32, p_drop=0.10)
        self.pool1  = nn.MaxPool2d((2,1))   # freq/2 only -> keep time

        self.block2 = AsymResBlock(32, 64, p_drop=0.12)
        self.pool2  = nn.MaxPool2d((2,2))   # freq/2 time/2

        self.block3 = AsymResBlock(64, 128, p_drop=0.15)
        self.pool3  = nn.MaxPool2d((2,1))   # freq/2 only

        self.block4 = AsymResBlock(128, 128, p_drop=0.18)
        self.pool4  = nn.MaxPool2d((2,1))   # freq/2 only

        # After pools: freq 128 -> 64 -> 32 -> 16 -> 8 (time ~ /2)
        # Freq-aware collapse (learned)
        self.freq_collapse = nn.Conv2d(128, 256, kernel_size=(8,1), bias=False)
        self.ln = nn.LayerNorm(256)

        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attn = nn.Sequential(
            nn.Linear(2*rnn_hidden, rnn_hidden),
            nn.Tanh(),
            nn.Linear(rnn_hidden, 1)
        )

        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2*rnn_hidden, n_classes)
        )

    def forward(self, x):
        if self.training:
            x = spec_augment(x, time_mask=32, freq_mask=16, p=1.0)

        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))  # [B,128,8,T']

        x = self.freq_collapse(x)       # [B,256,1,T']
        x = x.squeeze(2).transpose(1,2) # [B,T',256]
        x = self.ln(x)

        out, _ = self.rnn(x)            # [B,T',2H]
        a = torch.softmax(self.attn(out), dim=1)
        pooled = (out * a).sum(dim=1)  # [B,2H]

        return self.head(pooled)


if __name__ == "__main__":
    m = CRNNv2Light(n_classes=8)
    x = torch.randn(4,1,128,256)
    y = m(x)
    print(y.shape)
