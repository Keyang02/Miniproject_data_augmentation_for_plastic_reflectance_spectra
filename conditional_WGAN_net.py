import math, torch, torch.nn as nn
from torch.nn.utils import spectral_norm

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

class CondGen1D(nn.Module):
    def __init__(self, latent_dim=100, n_classes=7,
                 embed_dim=16, base_c=256, target_len=1301):
        super().__init__()
        self.target_len = target_len                         # 1301
        self.embed = nn.Embedding(n_classes, embed_dim)      # label → vector
        self.fc = nn.Linear(latent_dim + embed_dim, base_c * 81)

        self.net = nn.Sequential(                           # 81 → 162 → 324 → 648 → 1296
            UpsampleBlock(base_c,   base_c//2),             # 256 →128
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
    

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv1d(in_c, out_c, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.block(x)

class CondCritic1D(nn.Module):
    def __init__(self, n_classes=7, embed_dim=16,
                 base_c=64, input_len=1301):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)

        # first conv sees (1 + embed_dim) channels; label embedding is broadcast
        self.blocks = nn.Sequential(
            ConvBlock(1 + embed_dim, base_c),         # 1301 → 651
            ConvBlock(base_c, base_c*2),              # 651 → 326
            ConvBlock(base_c*2, base_c*4),            # 326 → 163
            ConvBlock(base_c*4, base_c*8),            # 163 → 82
        )
        flat_features = base_c*8 * math.ceil(input_len / 16)
        self.out = spectral_norm(nn.Linear(flat_features, 1))

    def forward(self, x, y):
        # x: (B,1,1301)   y: (B,)
        y_emb = self.embed(y).unsqueeze(2)                    # (B,embed_dim,1)
        y_img = y_emb.repeat(1, 1, x.size(2))                 # broadcast along length
        h = torch.cat([x, y_img], 1)
        h = self.blocks(h)
        h = h.view(h.size(0), -1)
        return self.out(h)                                    # (B,1) Wasserstein score

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
                 embed_dim: int = 16,
                 base_c: int = 256,
                 target_len: int = 1301):
        super().__init__()
        self.target_len = target_len  # kept for reference

        # 1️⃣  label embedding + fully‑connected projection to (B, base_c, 81)
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.fc = nn.Linear(latent_dim + embed_dim, base_c * 81)

        # 2️⃣  sequence of deconvolution blocks: 81 → 1296
        self.net = nn.Sequential(
            DeconvBlock(base_c, base_c // 2),   # 256 → 128, 81 → 162
            DeconvBlock(base_c // 2, base_c // 4),  # 162 → 324
            DeconvBlock(base_c // 4, base_c // 8),  # 324 → 648
            DeconvBlock(base_c // 8, base_c // 16), # 648 → 1296
            # final step: +5 samples ⇒ 1296 → 1301
            nn.ConvTranspose1d(base_c // 16, 1, kernel_size=6, stride=1, padding=0, bias=True),
            nn.Tanh(),
        )

    # ──────────────────────────────────────────────────────────────────
    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate a batch of spectra.

        Parameters
        ----------
        z : torch.Tensor (B, latent_dim)
            Latent noise.
        y : torch.Tensor (B,)
            Integer class labels.
        """
        y_emb = self.embed(y)                 # (B, embed_dim)
        x = torch.cat((z, y_emb), dim=1)      # (B, latent_dim+embed_dim)
        x = self.fc(x)                        # (B, base_c*81)
        x = x.view(x.size(0), -1, 81)         # (B, base_c, 81)
        return self.net(x)                    # (B, 1, 1301)