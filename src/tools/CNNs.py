import torch.nn as nn
import torch



class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),

            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.net(x).squeeze(-1).squeeze(-1)  # [B,128]
        return self.fc(x)

class AttnPool(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.w = nn.Linear(C, 1)

    def forward(self, x): # x [B,C,T]
        a = torch.softmax(self.w(x.transpose(1,2)), dim=1)  # [B,T,1]
        return (x.transpose(1,2) * a).sum(dim=1)            # [B,C]



class AttentionCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.GroupNorm(8, 16), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.1),

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
        )
        self.pool = AttnPool(128)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = self.feat(x)  # [B,128,F',T']
        x = x.mean(dim=2)  # [B,128,T']
        x = self.pool(x)  # [B,128]
        return self.fc(x)

def spec_augment(x, time_mask=24, freq_mask=12, p=0.5):
    """
    x: [B,1,128,T]
    Apply random time/freq masking on-the-fly.
    """
    if not x.requires_grad:  # allow use under no_grad too
        x = x.clone()

    B, _, F, T = x.shape
    if torch.rand(1).item() > p:
        return x

    # freq mask
    f = torch.randint(0, freq_mask + 1, (1,)).item()
    if f > 0:
        f0 = torch.randint(0, max(1, F - f), (1,)).item()
        x[:, :, f0:f0 + f, :] = 0.0

    # time mask
    t = torch.randint(0, time_mask + 1, (1,)).item()
    if t > 0:
        t0 = torch.randint(0, max(1, T - t), (1,)).item()
        x[:, :, :, t0:t0 + t] = 0.0

    return x

class CRNN(nn.Module):
    """
    Input: x [B, 1, 128, T] with T <= 256 (mel bins=128)
    Output: logits [B, n_classes]
    """
    def __init__(self, n_classes: int, rnn_hidden: int = 128, rnn_layers: int = 1):
        super().__init__()

        # CNN front-end (keeps time resolution reasonably high)
        self.cnn = nn.Sequential(
            # [B,1,128,T]
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),          # -> [B,32,64,T/2]
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),          # -> [B,64,32,T/4]
            nn.Dropout(0.15),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),          # freq /2 only -> [B,128,16,T/4]
            nn.Dropout(0.2),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),          # -> [B,128,8,T/4]
            nn.Dropout(0.25),
        )

        # After CNN: freq=8, channels=128 => feature size per time step = 128*8=1024
        self.rnn_in = 128 * 8
        self.rnn = nn.GRU(
            input_size=self.rnn_in,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if rnn_layers > 1 else 0.0,
        )

        self.attn = nn.Linear(2 * rnn_hidden, 1)
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2 * rnn_hidden, n_classes)
        )

    def forward(self, x):
        # x: [B,1,128,T]
        if self.training:
            x = spec_augment(x, time_mask=32, freq_mask=16, p=1.0)

        x = self.cnn(x)                   # [B,C,F,T']
        B, C, Freq, Tp = x.shape

        # flatten freq+channels, keep time
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,T',C,F]
        x = x.view(B, Tp, C * Freq)            # [B,T',1024]

        out, _ = self.rnn(x)              # [B,T',2H]

        # attention pooling over time
        a = torch.softmax(self.attn(out), dim=1)   # [B,T',1]
        pooled = (out * a).sum(dim=1)              # [B,2H]

        return self.fc(pooled)


class CRNN_V2(nn.Module):
    """
    Input: x [B, 1, 128, T] with T <= 256 (mel bins=128)
    Output: logits [B, n_classes]
    """
    def __init__(
        self,
        n_classes:int,
        rnn_hidden: int=128,
        rnn_layers: int=1,
        proj_dim: int=128,
        time_mask: int=32,
        freq_mask: int=16,
        spec_p: int=0.7,

    ):
        super().__init__()

        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.spec_p = spec_p

        # CNN front-end
        self.cnn = nn.Sequential(
            # [B,1,128,T]
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),       # -> [B,32,64,T/2]
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),       # -> [B,64,32,T/4]
            nn.Dropout(0.15),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),       # -> [B,128,16,T/4]
            nn.Dropout(0.2),

            nn.Conv2d(128, 160, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),       # -> [B,160,8,T/4]
            nn.Dropout(0.25),
        )

        # After CNN: freq=8, channels=160 => feature size per time step = 160*8=1280
        self.cnn_out_channels = 160
        self.cnn_out_freq = 8
        self.rnn_in = self.cnn_out_channels * self.cnn_out_freq  # 1280

        # Projection before GRU
        self.proj = nn.Sequential(
            nn.LayerNorm(self.rnn_in),
            nn.Linear(self.rnn_in, proj_dim),
            nn.ReLU(inplace=True),
        )

        # BiGRU backend
        self.rnn = nn.GRU(
            input_size=proj_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if rnn_layers > 1 else 0.0,
        )

        # MLP Attention over time
        self.attn = nn.Sequential(
            nn.Linear(2 * rnn_hidden, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2 * rnn_hidden, n_classes),
        )

    def forward(self, x):
        # x: [B,1,128,T]
        if self.training and self.spec_p > 0.0:
            x = spec_augment(
                x,
                time_mask=self.time_mask,
                freq_mask=self.freq_mask,
                p=self.spec_p,
            )

        x = self.cnn(x)  # [B,C,F,T']
        B, C, Freq, Tp = x.shape

        # flatten freq+channels, keep time: [B,T',C*F]
        x = x.permute(0, 3, 1, 2).contiguous()   # [B,T',C,F]
        x = x.view(B, Tp, C * Freq)              # [B,T',1280]

        # projection
        x = self.proj(x)                         # [B,T',proj_dim]

        # BiGRU
        out, _ = self.rnn(x)                     # [B,T',2*H]

        # Attention pooling over time
        a = torch.softmax(self.attn(out), dim=1) # [B,T',1]
        pooled = (out * a).sum(dim=1)            # [B,2*H]

        return self.fc(pooled)                   # [B,n_classes]


MODEL_PARAMS = {"SmallCNN": SmallCNN, "CRNN": CRNN, "CRNN_V2": CRNN_V2}