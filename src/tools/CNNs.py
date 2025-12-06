import torch.nn as nn
import torch

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

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

class SmallCNN(nn.Module):
    def __init__(self, n_classes, augment:bool = False):
        super().__init__()
        self.augment = augment
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
        if self.augment and self.training:
            x = spec_augment(x, time_mask=32, freq_mask=16, p=0.5)
        x = self.net(x).squeeze(-1).squeeze(-1)  # [B,128]
        return self.fc(x)

class SmallCRNN(nn.Module):
    def __init__(self, n_classes, hidden_size=128, n_layers=1, augment: bool = False):
        super().__init__()
        self.augment = augment

        # CNN : je garde ta structure mais j'ajoute BatchNorm et je réduis un peu les dropout
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.15),

            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.15),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )

        # Si ton F initial = 128 → après 4 MaxPool2d(2) : F_out = 128 / 16 = 8
        self.freq_out = 128 // 16
        self.rnn_input_size = 128 * self.freq_out

        self.rnn = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(2 * hidden_size, n_classes)

    def forward(self, x):
        # x: [B,1,128,T]
        if self.augment:
            x = spec_augment(x, time_mask=24, freq_mask=12, p=0.5)

        x = self.cnn(x)           # [B, 128, F_out, T_out]
        B, C, F, T = x.shape

        x = x.permute(0, 3, 1, 2)  # [B, T, C, F]
        x = x.reshape(B, T, C * F) # [B, T, C*F]

        out, h = self.rnn(x)       # out: [B, T, 2H], h: [2*n_layers, B, H]

        # On prend le dernier état de chaque direction et on concatène
        # h[-2] = dernier layer, direction forward ; h[-1] = dernier layer, direction backward
        h_fw = h[-2]  # [B, H]
        h_bw = h[-1]  # [B, H]
        h_cat = torch.cat([h_fw, h_bw], dim=1)  # [B, 2H]

        logits = self.fc(h_cat)
        return logits


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # vecteur de requête global appris (un "token résumé")
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            batch_first=True
        )

    def forward(self, x):
        """
        x: [B, T, D]
        retourne: [B, D]
        """
        B, T, D = x.shape
        q = self.query.expand(B, 1, D)  # [B,1,D] même query pour tous les batchs

        # q = "résumé" qui regarde toute la séquence x
        out, _ = self.attn(q, x, x)     # out: [B,1,D]
        return out.squeeze(1)           # [B,D]

class SmallCNN_Attn(nn.Module):
    def __init__(self, n_classes, augment: bool = False):
        super().__init__()
        self.augment = augment

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.15),

            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.15),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )

        # F initial = 128 → après 4 MaxPool2d(2) : F_out = 128 / 16 = 8
        self.freq_out = 128 // 16      # = 8 si F=128
        self.feature_dim = 128 * self.freq_out  # D = 128 * 8 = 1024

        self.attn_pool = TemporalAttentionPool(self.feature_dim)
        self.fc = nn.Linear(self.feature_dim, n_classes)

    def forward(self, x):
        # x: [B,1,128,T]
        if self.augment:
            x = spec_augment(x, time_mask=24, freq_mask=12, p=0.5)

        x = self.cnn(x)            # [B,128,F_out,T_out]
        B, C, F, T = x.shape

        # → [B, T_out, C*F]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B,T,C,F]
        x = x.view(B, T, C * F)                 # [B,T,D]

        x = self.attn_pool(x)                   # [B,D]
        logits = self.fc(x)                     # [B,n_classes]
        return logits


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



class CRNN(nn.Module):
    """
    Input: x [B, 1, 128, T] with T <= 256 (mel bins=128)
    Output: logits [B, n_classes]
    """
    def __init__(self, n_classes: int, rnn_hidden: int = 128, rnn_layers: int = 1, augment:bool= False):
        super().__init__()
        self.augment = augment
        # CNN front-end (keeps time resolution reasonably high)
        self.cnn = nn.Sequential(
            # [B,1,128,T]
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),          # -> [B,32,64,T/2]
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),          # -> [B,64,32,T/4]
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),          # freq /2 only -> [B,128,16,T/4]
            nn.Dropout(0.3),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),          # -> [B,128,8,T/4]
            nn.Dropout(0.3),
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
        if self.augment and self.training:
            x = spec_augment(x, time_mask=32, freq_mask=16, p=0.5)

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



MODEL_PARAMS = {"SmallCNN": SmallCNN, "CRNN": CRNN, "SmallCRNN": SmallCRNN, "SmallCNN_Attn": SmallCNN_Attn}