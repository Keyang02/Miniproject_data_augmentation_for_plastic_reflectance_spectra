import torch, torch.nn as nn
from torch.nn.utils import spectral_norm

def init_weights(m):
    """Initialize weights of the model."""
    classname = m.__class__.__name__

    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, 0.02)

def output_length_conv1D(input_len, kernel_size, stride=1, padding=0):
    """Calculate the output length of a 1D convolution."""
    return (input_len - kernel_size + 2 * padding) // stride + 1

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.block(x)

class CondCritic1D(nn.Module):
    def __init__(self, n_classes=7, embed_dim=8,
                 base_c=64, input_len=1301):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)

        k = input_len + embed_dim  
        for _ in range(4):
            k = output_length_conv1D(k, 4, stride=2, padding=1)
        kernel_size_last = k

        self.blocks = nn.Sequential(
            nn.Conv1d(1, base_c, 4, stride=2, padding=1),   # L:1309 -> 654  C:1 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            ConvBlock(base_c, base_c*2),                    # L:654 -> 327   C:64 -> 128
            ConvBlock(base_c*2, base_c*4),                  # L:327 -> 163   C:128 -> 256
            ConvBlock(base_c*4, base_c*8),                  # L:163 -> 81    C:256 -> 512
            nn.Dropout(0.1),                                # L:81 -> 81
            nn.Conv1d(base_c * 8, 1, kernel_size=kernel_size_last, stride=1, padding=0, bias=False)  # L:81 -> 1   C:512 -> 1
        )
        # flat_features = base_c * 8 * kernel_size_last  # 512 * 83
        # self.out = spectral_norm(nn.Linear(flat_features, 1))
        

    def forward(self, x, y):
        # x: (B,1,1301)   y: (B,)
        y_emb = self.embed(y).unsqueeze(1)                    # (B,1,embed_dim)
        h = torch.cat([x, y_emb], 2)                     # (B,1,1301+embed_dim)
        h = self.blocks(h)
        # h = h.view(h.size(0), -1)
        return h.view(h.size(0), -1)                                     # (B,1) Wasserstein score
    
class CondCritic1D_2labels(nn.Module):
    def __init__(self, n_classes=7, embed_dim1=8, embed_dim2=4,
                 base_c=64, input_len=1301):
        super().__init__()
        self.embed1 = nn.Embedding(n_classes, embed_dim1)
        self.embed2 = nn.Embedding(n_classes, embed_dim2)

        k = input_len + embed_dim1 + embed_dim2
        for _ in range(4):
            k = output_length_conv1D(k, 4, stride=2, padding=1)
        kernel_size_last = k

        self.blocks = nn.Sequential(
            nn.Conv1d(1, base_c, 4, stride=2, padding=1),   # L:1309 -> 654  C:1 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            ConvBlock(base_c, base_c*2),                    # L:654 -> 327   C:64 -> 128
            ConvBlock(base_c*2, base_c*4),                  # L:327 -> 163   C:128 -> 256
            ConvBlock(base_c*4, base_c*8),                  # L:163 -> 81    C:256 -> 512
            nn.Dropout(0.1),                                # L:81 -> 81
            nn.Conv1d(base_c * 8, 1, kernel_size=kernel_size_last, stride=1, padding=0, bias=False)  # L:81 -> 1   C:512 -> 1
        )
        # flat_features = base_c * 8 * kernel_size_last  # 512 * 83
        # self.out = spectral_norm(nn.Linear(flat_features, 1))
        

    def forward(self, x, y1, y2):
        # x: (B,1,1301)   y: (B,)
        y1_emb = self.embed1(y1).unsqueeze(1)                    # (B,1,embed_dim)
        y2_emb = self.embed2(y2).unsqueeze(1)                    # (B,1,embed_dim)
        h = torch.cat([x, y1_emb, y2_emb], 2)                     # (B,1,1301+embed_dim*2)
        h = self.blocks(h)
        # h = h.view(h.size(0), -1)
        return h.view(h.size(0), -1)                                     # (B,1) Wasserstein score

class UpsampleBlock(nn.Module):
    """nearest-neighbor ↑2  +  Conv1d (padding='same')"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(True),
        )
    def forward(self, x): return self.block(x)

class CondGen1D_upsample(nn.Module):
    def __init__(self, latent_dim=100, n_classes=7,
                 embed_dim=8, base_c=256, target_len=1301):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)      # label → vector
        self.fc = nn.Linear(latent_dim + embed_dim, base_c * 81)

        self.net = nn.Sequential(                           # 81 → 162 → 324 → 648 → 1296
            UpsampleBlock(base_c,   base_c//2),            # 256 →128
            UpsampleBlock(base_c//2, base_c//4),            # 128 → 64
            UpsampleBlock(base_c//4, base_c//8),            # 64  → 32
            UpsampleBlock(base_c//8, base_c//16),           # 32  → 16
            nn.Upsample(size=target_len, mode="linear", align_corners=False),
            nn.Conv1d(base_c//16, 1, kernel_size=3, padding=1),
            nn.Tanh(),                                      # output in (-1, 1)
        )

    def forward(self, z, y):
        y_emb = self.embed(y)
        x = torch.cat((z, y_emb), 1)
        x = self.fc(x).view(x.size(0), -1, 81)
        return self.net(x)          # (B, 1, 1301)

class DeconvBlock(nn.Module):
    """(ConvTranspose1d ↑2 → BN → ReLU)"""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            # kernel=4, stride=2, padding=1 doubles the length exactly
            nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CondGen1D_ConvT(nn.Module):
    """Conditional 1-D generator using transposed convolutions for up-sampling."""

    def __init__(self,
                 latent_dim: int = 100,
                 n_classes: int = 7,
                 embed_dim: int = 8,
                 base_c: int = 256,
                 target_len: int = 1301):
        super().__init__()
        self.target_len = target_len  # kept for reference

        # label embedding + fully‑connected projection to (B, base_c, 81)
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.fc = nn.Linear(latent_dim + embed_dim, base_c * 81)

        # sequence of deconvolution blocks: 81 -> 1296
        self.net = nn.Sequential(
            DeconvBlock(base_c, base_c // 2),       # C:256 -> 128, L:81 -> 162
            DeconvBlock(base_c // 2, base_c // 4),  # C:128 -> 64,  L:162 -> 324
            DeconvBlock(base_c // 4, base_c // 8),  # C:64  -> 32,  L:324 -> 648
            DeconvBlock(base_c // 8, base_c // 16), # C:32  -> 16,  L:648 -> 1296
            # final step: +5 samples ⇒ 1296 → 1301
            nn.ConvTranspose1d(base_c // 16, 1, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    # ──────────────────────────────────────────────────────────────────
    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate a batch of spectra."""
        # z: (B, latent_dim)   y: (B,)
        # z: latent noise vector, y: class label
        y_emb = self.embed(y)                 # (B, embed_dim)
        x = torch.cat((z, y_emb), dim=1)      # (B, latent_dim+embed_dim)
        x = self.fc(x)                        # (B, base_c*81)
        x = x.view(x.size(0), -1, 81)         # (B, base_c, 81)
        return self.net(x)                    # (B, 1, 1301)

class CondGen1D_ConvT_2labels(nn.Module):
    """Conditional 1-D generator using transposed convolutions for up-sampling."""

    def __init__(self,
                 latent_dim: int = 100,
                 n_classes: int = 7,
                 embed_dim1: int = 8,
                 embed_dim2: int = 4,
                 base_c: int = 256,
                 target_len: int = 1301):
        super().__init__()
        self.target_len = target_len  # kept for reference

        # label embedding + fully‑connected projection to (B, base_c, 81)
        self.embed1 = nn.Embedding(n_classes, embed_dim1)
        self.embed2 = nn.Embedding(n_classes, embed_dim2)
        self.fc = nn.Linear(latent_dim + embed_dim1 + embed_dim2, base_c * 81)

        # sequence of deconvolution blocks: 81 -> 1296
        self.net = nn.Sequential(
            DeconvBlock(base_c, base_c // 2),       # C:256 -> 128, L:81 -> 162
            DeconvBlock(base_c // 2, base_c // 4),  # C:128 -> 64,  L:162 -> 324
            DeconvBlock(base_c // 4, base_c // 8),  # C:64  -> 32,  L:324 -> 648
            DeconvBlock(base_c // 8, base_c // 16), # C:32  -> 16,  L:648 -> 1296
            # final step: +5 samples ⇒ 1296 → 1301
            nn.ConvTranspose1d(base_c // 16, 1, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    # ──────────────────────────────────────────────────────────────────
    def forward(self, z: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Generate a batch of spectra."""
        # z: (B, latent_dim)   y: (B,)
        # z: latent noise vector, y: class label
        y1_emb = self.embed1(y1)               # (B, embed_dim1)
        y2_emb = self.embed2(y2)               # (B, embed_dim2)
        x = torch.cat((z, y1_emb, y2_emb), dim=1)  # (B, latent_dim+embed_dim1+embed_dim2)
        x = self.fc(x)                        # (B, base_c*81)
        x = x.view(x.size(0), -1, 81)         # (B, base_c, 81)
        return self.net(x)                    # (B, 1, 1301)