import torch, torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

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
            # nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.block(x)

class CondCritic1D(nn.Module):
    def __init__(self, 
                 n_classes: int = 7, 
                 embed_dim: int = 8,
                 base_c: int = 64, 
                 input_len: int = 1301):
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
    def __init__(self, 
                 embed_classes1: int = 11, 
                 embed_dim1: int = 16, 
                 embed_classes2: int = 4,
                 embed_dim2: int = 8,
                 base_c: int = 64, 
                 input_len: int = 1301,
                 isCR: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.embed1 = nn.Embedding(embed_classes1, embed_dim1)
        self.embed2 = nn.Embedding(embed_classes2, embed_dim2)
        self.isCR = isCR

        k = input_len + embed_dim1 + embed_dim2
        for _ in range(4):
            k = output_length_conv1D(k, 4, stride=2, padding=1)
        kernel_size_last = k

        self.blocks = nn.Sequential(
            nn.Conv1d(1, base_c, 4, stride=2, padding=1),   # L:1309 -> 654  C:1 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            ConvBlock(base_c, base_c*2),                    # L:654 -> 327   C:64 -> 128
            ConvBlock(base_c*2, base_c*4),                  # L:327 -> 163   C:128 -> 256
            ConvBlock(base_c*4, base_c*8)                  # L:163 -> 81    C:256 -> 512
        )  
        self.out = nn.Sequential(
            nn.Dropout(dropout),                                # L:81 -> 81
            nn.Conv1d(base_c * 8, 1, kernel_size=kernel_size_last, stride=1, padding=0, bias=False)  # L:81 -> 1   C:512 -> 1
        )

    def forward(self, x, y1, y2):
        # x: (B,1,1301)   y: (B,)
        y1_emb = self.embed1(y1).unsqueeze(1)                    # (B,1,embed_dim)
        y2_emb = self.embed2(y2).unsqueeze(1)                    # (B,1,embed_dim)
        v = torch.cat([x, y1_emb, y2_emb], 2)                     # (B,1,1301+embed_dim*2)
        v = self.blocks(v)
        h = self.out(v)
        if self.isCR:
            return h.view(h.size(0), -1), v.view(v.size(0), -1)  # (B,1) Wasserstein score, (B,512*83) features
        else:  
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
            nn.InstanceNorm1d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
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
                 embed_classes1: int = 11,
                 embed_dim1: int = 16,
                 embed_classes2: int = 4,
                 embed_dim2: int = 8,
                 base_c: int = 256,
                 target_len: int = 1301):
        super().__init__()
        self.target_len = target_len  # kept for reference

        # label embedding + fully‑connected projection to (B, base_c, 81)
        self.embed1 = nn.Embedding(embed_classes1, embed_dim1)
        self.embed2 = nn.Embedding(embed_classes2, embed_dim2)
        self.fc = nn.Linear(latent_dim + embed_dim1 + embed_dim2, base_c * 81)

        # sequence of deconvolution blocks: 81 -> 1296
        self.net = nn.Sequential(
            DeconvBlock(base_c, base_c // 2),       # C:256 -> 128, L:81 -> 162
            DeconvBlock(base_c // 2, base_c // 4),  # C:128 -> 64,  L:162 -> 324
            DeconvBlock(base_c // 4, base_c // 8),  # C:64  -> 32,  L:324 -> 648
            DeconvBlock(base_c // 8, base_c // 16), # C:32  -> 16,  L:648 -> 1296
            # final step: +5 samples ⇒ 1296 → 1301
            nn.ConvTranspose1d(base_c // 16, 16, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Conv1d(16, 1, kernel_size=5, padding=2),   # smoothing conv
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
    

class UpsampleBlock2(nn.Module):
    """(Upsample ↑2 → Conv1d → Norm → Activation)"""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
            nn.Conv1d(in_c, out_c, kernel_size=5, stride=1, padding=2, bias=False),
            nn.InstanceNorm1d(out_c, affine=True),   # smoother than BatchNorm
            nn.LeakyReLU(0.2, inplace=True)          # smoother than ReLU
        )

    def forward(self, x):
        return self.block(x)


class CondGen1D_Upsample_2labels(nn.Module):
    """Conditional 1-D generator using Upsample+Conv1d for smooth spectra."""

    def __init__(self,
                 latent_dim: int = 100,
                 embed_classes1: int = 11,
                 embed_dim1: int = 16,
                 embed_classes2: int = 4,
                 embed_dim2: int = 8,
                 base_c: int = 256,
                 target_len: int = 1301):
        super().__init__()
        self.target_len = target_len

        # embeddings
        self.embed1 = nn.Embedding(embed_classes1, embed_dim1)
        self.embed2 = nn.Embedding(embed_classes2, embed_dim2)

        # project latent+labels into a short sequence
        self.fc = nn.Linear(latent_dim + embed_dim1 + embed_dim2, base_c * 81)

        # upsampling path (81 → ~1296)
        self.net = nn.Sequential(
            UpsampleBlock2(base_c, base_c // 2),      # 81 → 162
            UpsampleBlock2(base_c // 2, base_c // 4), # 162 → 324
            UpsampleBlock2(base_c // 4, base_c // 8), # 324 → 648
            UpsampleBlock2(base_c // 8, base_c // 16) # 648 → 1296
        )

        # final conv to reach exactly 1301
        self.final = nn.Sequential(
            nn.Conv1d(base_c // 16, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

        def _init_film(m):
            if isinstance(m, FiLM1D):
                nn.init.zeros_(m.to_gamma.weight); nn.init.zeros_(m.to_gamma.bias)
                nn.init.zeros_(m.to_beta.weight); nn.init.zeros_(m.to_beta.bias)

        self.apply(_init_film)


    def forward(self, z: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        # concatenate latent + embeddings
        y1_emb = self.embed1(y1)
        y2_emb = self.embed2(y2)
        x = torch.cat((z, y1_emb, y2_emb), dim=1)

        # project and reshape
        x = self.fc(x)
        x = x.view(x.size(0), -1, 81)   # (B, base_c, 81)

        # upsample to near target length
        x = self.net(x)

        # adjust to exact target length (1301)
        x = self.final(x)
        if x.size(-1) != self.target_len:
            x = nn.functional.interpolate(x, size=self.target_len, mode="linear", align_corners=True)

        return x

class FiLM1D(nn.Module):
    """Feature-wise linear modulation for 1D activations: y = x * (1 + γ) + β."""
    def __init__(self, n_channels: int, cond_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, n_channels)
        self.to_beta  = nn.Linear(cond_dim, n_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B,C,L), cond: (B,cond_dim)
        gamma = self.to_gamma(cond).unsqueeze(-1)  # (B,C,1)
        beta  = self.to_beta(cond).unsqueeze(-1)   # (B,C,1)
        return x * (1 + gamma) + beta

class CondGen1D_Upsample_FiLM(nn.Module):
    """Conditional 1-D generator with Upsample+Conv and FiLM conditioning at each stage."""
    def __init__(self,
                 latent_dim: int = 100,
                 embed_classes1: int = 11,
                 embed_dim1: int = 16,
                 embed_classes2: int = 4,
                 embed_dim2: int = 8,
                 base_c: int = 256,
                 target_len: int = 1301):
        super().__init__()
        self.target_len = target_len

        # ── label embeddings
        self.embed1 = nn.Embedding(embed_classes1, embed_dim1)
        self.embed2 = nn.Embedding(embed_classes2, embed_dim2)
        cond_dim = embed_dim1 + embed_dim2

        # ── project latent + labels to a short sequence (length 81)
        self.fc = nn.Linear(latent_dim + embed_dim1 + embed_dim2, base_c * 81)

        # ── upsampling path (81 → ~1296) as separate modules so we can FiLM after each
        self.up1 = UpsampleBlock2(base_c,       base_c // 2)   # 81 → 162
        self.up2 = UpsampleBlock2(base_c // 2,  base_c // 4)   # 162 → 324
        self.up3 = UpsampleBlock2(base_c // 4,  base_c // 8)   # 324 → 648
        self.up4 = UpsampleBlock2(base_c // 8,  base_c // 16)  # 648 → 1296

        # ── FiLM adapters (one per stage)
        self.film1 = FiLM1D(base_c // 2,  cond_dim)
        self.film2 = FiLM1D(base_c // 4,  cond_dim)
        self.film3 = FiLM1D(base_c // 8,  cond_dim)
        self.film4 = FiLM1D(base_c // 16, cond_dim)

        # ── final conv to hit exactly target_len
        self.final = nn.Sequential(
            nn.Conv1d(base_c // 16, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        # Concatenate latent + embeddings
        e1 = self.embed1(y1)         # (B, embed_dim1)
        e2 = self.embed2(y2)         # (B, embed_dim2)
        cond = torch.cat([e1, e2], dim=1)

        x = torch.cat([z, e1, e2], dim=1)      # (B, latent + emb1 + emb2)
        x = self.fc(x).view(x.size(0), -1, 81) # (B, base_c, 81)

        # Upsample with FiLM after each stage
        x = self.up1(x); x = self.film1(x, cond)
        x = self.up2(x); x = self.film2(x, cond)
        x = self.up3(x); x = self.film3(x, cond)
        x = self.up4(x); x = self.film4(x, cond)

        # Adjust to exact length and range
        x = self.final(x)
        if x.size(-1) != self.target_len:
            x = nn.functional.interpolate(x, size=self.target_len, mode="linear", align_corners=True)
        return x
       
class UpsampleBlock2FiLM(nn.Module):
    """(Upsample ↑2 → Conv1d → InstanceNorm(affine=False) → FiLM → LeakyReLU)"""
    def __init__(self, in_c: int, out_c: int, cond_dim: int):
        super().__init__()
        self.ups = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=5, stride=1, padding=2, bias=False)
        self.norm = nn.InstanceNorm1d(out_c, affine=False)  # FiLM supplies affine
        self.film = FiLM1D(out_c, cond_dim)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

        # make FiLM start as identity
        nn.init.zeros_(self.film.to_gamma.weight); nn.init.zeros_(self.film.to_gamma.bias)
        nn.init.zeros_(self.film.to_beta.weight);  nn.init.zeros_(self.film.to_beta.bias)

    def forward(self, x, cond):
        x = self.ups(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, cond)   # γ,β applied here
        x = self.act(x)
        return x

class CondGen1D_Upsample_FiLM_Optimized(nn.Module):
    def __init__(self, latent_dim=100,
                 embed_classes1=11, embed_dim1=16,
                 embed_classes2=4,  embed_dim2=8,
                 base_c=256, target_len=1301):
        super().__init__()
        self.target_len = target_len

        self.embed1 = nn.Embedding(embed_classes1, embed_dim1)
        self.embed2 = nn.Embedding(embed_classes2, embed_dim2)
        cond_dim = embed_dim1 + embed_dim2

        self.fc = nn.Linear(latent_dim + embed_dim1 + embed_dim2, base_c * 81)

        self.up1 = UpsampleBlock2FiLM(base_c,       base_c // 2,  cond_dim)
        self.up2 = UpsampleBlock2FiLM(base_c // 2,  base_c // 4,  cond_dim)
        self.up3 = UpsampleBlock2FiLM(base_c // 4,  base_c // 8,  cond_dim)
        self.up4 = UpsampleBlock2FiLM(base_c // 8,  base_c // 16, cond_dim)

        self.final = nn.Sequential(
            nn.Conv1d(base_c // 16, 1, kernel_size=7, padding=3, bias=True),
            nn.Tanh()
        )

    def forward(self, z, y1, y2):
        e1 = self.embed1(y1)
        e2 = self.embed2(y2)
        cond = torch.cat([e1, e2], dim=1)

        x = torch.cat([z, e1, e2], dim=1)
        x = self.fc(x).view(x.size(0), -1, 81)

        x = self.up1(x, cond)
        x = self.up2(x, cond)
        x = self.up3(x, cond)
        x = self.up4(x, cond)

        x = self.final(x)
        if x.size(-1) != self.target_len:
            x = F.interpolate(x, size=self.target_len, mode="linear", align_corners=True)
        return x

class CondCritic1D_PD_Stable(nn.Module):
    def __init__(self, embed_classes1=11, embed_dim1=16,
                 embed_classes2=4,  embed_dim2=8,
                 base_c=64, input_len=1301, isCR=False, dropout=0.1):
        super().__init__()
        self.isCR = isCR
        C_feat = base_c * 8

        # label embeddings
        self.embed1 = nn.Embedding(embed_classes1, embed_dim1)
        self.embed2 = nn.Embedding(embed_classes2, embed_dim2)

        # feature extractor (same downsamples as before)
        self.blocks = nn.Sequential(
            nn.Conv1d(1, base_c, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_c, base_c*2, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_c*2, base_c*4, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_c*4, base_c*8, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)    # length-independent, better with GP

        # projection head (normalized dot product) + unconditional score
        self.proj = nn.Linear(embed_dim1 + embed_dim2, C_feat, bias=False)
        self.fc   = nn.Linear(C_feat, 1)
        self.scale = nn.Parameter(torch.tensor(1.0))  # learnable scale for proj term

    def forward(self, x, y1, y2):
        v = self.blocks(x)              # (B, C_feat, L')
        v = self.dropout(v)
        h_map = v                       # keep map for CR if needed
        h = self.pool(v).squeeze(-1)    # (B, C_feat)

        # projection term
        e = torch.cat([self.embed1(y1), self.embed2(y2)], dim=1)  # (B, d1+d2)
        e = self.proj(e)                                          # (B, C_feat)

        # normalized dot product to prevent scale blow-up
        h_n = F.normalize(h, dim=1)
        e_n = F.normalize(e, dim=1)
        proj = (h_n * e_n).sum(dim=1, keepdim=True) * self.scale

        score = self.fc(h) + proj

        if self.isCR:
            feats = h_map.mean(dim=2)  # fixed (B, C_feat) feature for CR
            return score, feats
        else:
            return score