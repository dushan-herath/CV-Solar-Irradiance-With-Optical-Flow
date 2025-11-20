import math
import torch
from torch import nn
import timm
import random
# =========================================
# IMAGE ENCODER
# =========================================
class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = 'vit_base_patch16_224', pretrained: bool = True,
                 freeze: bool = True, unfreeze_last: int = 0):
        """
        Args:
            model_name: timm model name
            pretrained: load pretrained weights
            freeze: freeze backbone parameters
            unfreeze_last: number of last layers/stages to unfreeze
        """
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self.out_dim = self.backbone.num_features

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

            if unfreeze_last > 0:
                self._unfreeze_last_layers(unfreeze_last)

    def _unfreeze_last_layers(self, n: int):
        backbone_type = self.backbone.__class__.__name__.lower()

        if 'swin' in backbone_type:
            # Swin: layers[0..3] are the 4 stages
            layers = [self.backbone.layers[0], self.backbone.layers[1],
                      self.backbone.layers[2], self.backbone.layers[3]]
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            # Optionally unfreeze final norm
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        elif 'resnet' in backbone_type:
            # ResNet: layer1..layer4
            layers = [self.backbone.layer1, self.backbone.layer2,
                      self.backbone.layer3, self.backbone.layer4]
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        else:
            print(f"Unfreeze last layers: please customize for backbone {backbone_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
# =========================================
# TIME SERIES ENCODER
# =========================================
class TS_Encoder(nn.Module):
    def __init__(self, ts_feat_dim: int, ts_embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.1):
        """
        Args:
            ts_feat_dim: number of input TS features per time step
            ts_embed_dim: output embedding dimension
            hidden_dim: hidden dimension of intermediate layer
            dropout: dropout rate
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ts_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, ts_embed_dim),
            nn.GELU(),
            nn.LayerNorm(ts_embed_dim),
            nn.Dropout(dropout),
        )
        self.out_dim = ts_embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) - batch, time steps, features
        Returns:
            (B, T, ts_embed_dim)
        """
        return self.proj(x)


# =========================================
# POSITIONAL ENCODING
# =========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


# =========================================
# FUSION MODULE
# =========================================
class GatedFusion(nn.Module):
    def __init__(self, img_dim, ts_dim, fused_dim, dropout: float = 0.1):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, fused_dim)
        self.ts_proj = nn.Linear(ts_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(fused_dim*2, fused_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feats, ts_feats):
        # Align lengths (use last T_img TS steps)
        ts_last = ts_feats[:, -img_feats.shape[1]:, :]
        
        img_proj = self.img_proj(img_feats)
        ts_proj = self.ts_proj(ts_last)

         # -------------------------------
        # L2 magnitude matching (new)
        # -------------------------------
        norm_img = img_proj.norm(dim=-1, keepdim=True)
        norm_ts  = ts_proj.norm(dim=-1, keepdim=True)
        scale = (norm_ts + 1e-6) / (norm_img + 1e-6)
        img_proj = img_proj * scale.detach()  

        if random.random() < 0.2:
            print(f"img_feats norm: {img_proj.norm(dim=-1).mean().item():.3f}, "
              f"ts_feats norm: {ts_proj.norm(dim=-1).mean().item():.3f}")
            
        gate = self.gate(torch.cat([img_proj, ts_proj], dim=-1))

        fused = gate * img_proj + (1 - gate) * ts_proj
        return self.dropout(fused)


# =========================================
# TEMPORAL TRANSFORMER
# =========================================
class FusionTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.transformer(x)
        return x


# =========================================
# MULTIMODAL FORECASTER
# =========================================
class MultimodalForecaster(nn.Module):
    def __init__(
        self,
        sky_encoder: ImageEncoder,
        flow_encoder: ImageEncoder,
        ts_feat_dim: int,
        img_embed_dim: int = None,
        ts_embed_dim: int = 128,
        fused_dim: int = 256,
        d_model: int = 256,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        horizon: int = 25,
        target_dim: int = 3,
    ):
        super().__init__()
        self.sky_encoder = sky_encoder
        self.flow_encoder = flow_encoder

        # Encoder output dims
        self.sky_img_dim = sky_encoder.out_dim
        self.flow_img_dim = flow_encoder.out_dim

        # Time-series encoder
        self.ts_encoder = TS_Encoder(ts_feat_dim=ts_feat_dim, ts_embed_dim=ts_embed_dim, dropout=dropout)

        # Positional encodings
        self.sky_pos_enc = PositionalEncoding(self.sky_img_dim)
        self.flow_pos_enc = PositionalEncoding(self.flow_img_dim)
        self.ts_pos_enc = PositionalEncoding(ts_embed_dim)

        # Fusion: fuse sky + flow + ts
        self.fusion = GatedFusion(img_dim=self.sky_img_dim + self.flow_img_dim, ts_dim=ts_embed_dim, fused_dim=fused_dim)

        # Temporal modeling
        self.temporal = FusionTransformer(
            input_dim=fused_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.horizon = horizon
        self.target_dim = target_dim

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon * target_dim)
        )

        # Attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, sky_imgs: torch.Tensor, flow_imgs: torch.Tensor, ts: torch.Tensor):
        B, T_img, C, H, W = sky_imgs.shape
        B, T_ts, F = ts.shape

        # Encode sky images
        sky_feats = self.sky_encoder(sky_imgs.view(B*T_img, C, H, W))
        sky_feats = sky_feats.view(B, T_img, -1)
        sky_feats = self.sky_pos_enc(sky_feats)

        # Encode optical flow images
        flow_feats = self.flow_encoder(flow_imgs.view(B*T_img, C, H, W))
        flow_feats = flow_feats.view(B, T_img, -1)
        flow_feats = self.flow_pos_enc(flow_feats)

        # Concatenate both image embeddings
        img_feats = torch.cat([sky_feats, flow_feats], dim=-1)

        # Encode time-series
        ts_feats = self.ts_encoder(ts)
        ts_feats = self.ts_pos_enc(ts_feats)

        

        # Fuse modalities
        fused_feats = self.fusion(img_feats, ts_feats)


            
        # Temporal transformer
        out_seq = self.temporal(fused_feats)

        # Attention pooling
        scores = self.attn_pool(out_seq)
        weights = torch.softmax(scores, dim=1)
        context = (weights * out_seq).sum(dim=1)

        # Predict
        out = self.head(context)
        out = out.view(B, self.horizon, self.target_dim)
        return out


# =========================================
# TEST SCRIPT
# =========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Image encoders
    sky_enc = ImageEncoder(model_name='vit_small_patch16_224', pretrained=True, freeze=True)
    flow_enc = ImageEncoder(model_name= 'resnet18', pretrained=True, freeze=True)

    # Model
    model = MultimodalForecaster(
        sky_encoder=sky_enc,
        flow_encoder=flow_enc,
        ts_feat_dim=5,
        ts_embed_dim=64,
        fused_dim=128,
        d_model=128,
        num_layers=2,
        nhead=4,
        horizon=25,
        target_dim=1
    ).to(device)

    # Dummy input
    B, T_img, T_ts = 2, 5, 30
    sky_imgs = torch.randn(B, T_img, 3, 224, 224).to(device)
    flow_imgs = torch.randn(B, T_img, 3, 224, 224).to(device)
    ts = torch.randn(B, T_ts, 5).to(device)

    preds = model(sky_imgs, flow_imgs, ts)
    print("preds.shape:", preds.shape)  # (B, horizon, target_dim)
