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
    if not x.requires_grad:
        x = x.clone()
    B, _, F, T = x.shape
    if torch.rand(1).item() > p:
        return x
    f = torch.randint(0, freq_mask + 1, (1,)).item()
    if f > 0:
        f0 = torch.randint(0, max(1, F - f), (1,)).item()
        x[:, :, f0:f0 + f, :] = 0.0
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
            nn.MaxPool2d(2), nn.Dropout(0.2),

            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
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
            x = spec_augment(x, time_mask=32, freq_mask=16, p=0.5)

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




MODEL_PARAMS = {"SmallCNN": SmallCNN, "SmallCRNN": SmallCRNN}