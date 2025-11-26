from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import toml
import cv2
import numpy as np
import xlrd

from training import Trainer
from validation import Validator
from test import *
from data import CMHADDataset
import torchvision.models as models


class SensorTransformerEncoder(nn.Module):

    def __init__(self,
                 input_dim=6,
                 embed_dim=128,
                 num_heads=4,
                 ff_dim=None,
                 num_layers=2,
                 dropout=0.1,
                 dim_feedforward=None,
                 max_len=128):
        super().__init__()


        if dim_feedforward is not None:
            ff_dim = dim_feedforward
        if ff_dim is None:
            ff_dim = embed_dim * 2

        self.input_proj = nn.Linear(input_dim, embed_dim)


        self.max_len = max_len
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_len, embed_dim)
        )  

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.input_proj(x)


        B, L, C = x.shape
        pos = self.pos_embed[:, :L, :]
        x = x + pos
        x = self.encoder(x)

        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)

        return x



class GMU2(nn.Module):

    def __init__(self, sensor_dim, video_dim, out_dim):
        super().__init__()
        self.sensor_fc = nn.Linear(sensor_dim, out_dim)
        self.video_fc  = nn.Linear(video_dim, out_dim)
        self.gate_fc   = nn.Linear(sensor_dim + video_dim, out_dim)

    def forward(self, sensor, video):

        h_s = torch.tanh(self.sensor_fc(sensor))  
        h_v = torch.tanh(self.video_fc(video))   


        h_cat = torch.cat([sensor, video], dim=1)  
        g = torch.sigmoid(self.gate_fc(h_cat))     


        z = g * h_s + (1.0 - g) * h_v             
        return z

class ConcatFusion(nn.Module):

    def __init__(self, sensor_dim, video_dim, out_dim):
        super().__init__()
        in_dim = sensor_dim + video_dim
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, sensor, video):
        h = torch.cat([sensor, video], dim=1)
        h = F.relu(self.bn1(self.fc1(h)))
        h = self.fc2(h)
        return h


class FiLMFusion(nn.Module):

    def __init__(self, sensor_dim, video_dim, out_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(sensor_dim, video_dim)
        self.beta_fc  = nn.Linear(sensor_dim, video_dim)
        self.out_fc   = nn.Linear(sensor_dim + video_dim, out_dim)
        self.out_bn   = nn.BatchNorm1d(out_dim)

    def forward(self, sensor, video):
        gamma = self.gamma_fc(sensor)
        beta  = self.beta_fc(sensor)

        mod_video = gamma * video + beta

        h = torch.cat([sensor, mod_video], dim=1)
        h = F.relu(self.out_bn(self.out_fc(h)))
        return h


class CrossAttentionFusion(nn.Module):

    def __init__(self, sensor_dim, video_dim, out_dim, num_heads=4):
        super().__init__()
        d_model = max(sensor_dim, video_dim)   
        self.sensor_proj = nn.Linear(sensor_dim, d_model)
        self.video_proj  = nn.Linear(video_dim, d_model)

        self.xattn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.out_fc = nn.Linear(d_model, out_dim)
        self.out_bn = nn.BatchNorm1d(out_dim)

    def forward(self, sensor, video):

        s = self.sensor_proj(sensor).unsqueeze(1)
        v = self.video_proj(video).unsqueeze(1)

        seq = torch.cat([s, v], dim=1)
        x, _ = self.xattn(seq, seq, seq)

        fused = x.mean(dim=1)
        fused = F.relu(self.out_bn(self.out_fc(fused)))
        return fused


class BilinearFusion(nn.Module):

    def __init__(self, sensor_dim, video_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.proj_s = nn.Linear(sensor_dim, hidden_dim)
        self.proj_v = nn.Linear(video_dim, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, out_dim)
        self.out_bn = nn.BatchNorm1d(out_dim)

    def forward(self, sensor, video):
        s = self.proj_s(sensor)
        v = self.proj_v(video)
        h = s * v
        h = F.relu(self.out_bn(self.out_fc(h)))
        return h 


class TSNMobileNetV2Encoder(nn.Module):

    def __init__(self,
                 out_dim=256,
                 use_temporal_transformer=True,
                 temporal_num_layers=2,
                 temporal_num_heads=4,
                 temporal_ff_dim=2048,
                 frame_chunk=8,
                 freeze_backbone=True):
        super().__init__()


        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        feat_dim = net.last_channel  


        self.backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d((1, 1)))
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.feat_dim = feat_dim
        self.frame_chunk = frame_chunk

        self.use_temporal_transformer = use_temporal_transformer
        if use_temporal_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=temporal_num_heads,
                dim_feedforward=temporal_ff_dim,
                dropout=0.1,
                batch_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=temporal_num_layers
            )
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x):

        B, C, T, H, W = x.shape


        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)


        if self.frame_chunk is None:
            feat = self.backbone(x)
            feats = []
            for i in range(0, B * T, self.frame_chunk):
                feats.append(self.backbone(x[i:i + self.frame_chunk]))
            feat = torch.cat(feats, dim=0)

        feat = feat.flatten(1)

        feat = feat.view(B, T, self.feat_dim)


        if self.use_temporal_transformer:

            feat = self.temporal_encoder(feat)

            feat = feat.transpose(1, 2)
            feat = self.temporal_pool(feat).squeeze(-1)
        else:

            feat = feat.mean(dim=1)


        feat = self.proj(feat)
        return feat


def build_video_encoder(name: str, out_dim=256):

    name = name.lower()
    if name == "tsn_mbv2":
        return TSNMobileNetV2Encoder(
            out_dim=out_dim,
            use_temporal_transformer=True,
            temporal_num_layers=2,
            temporal_num_heads=4,
            temporal_ff_dim=2048,
            frame_chunk=8,
            freeze_backbone=True
        )
    raise ValueError(f"unknown video encoder: {name}")



class Net(nn.Module):
    def __init__(self,
                 sensor_input_dim: int = 6,
                 video_feature_dim: int = 256,
                 fused_hidden: int = 128,
                 num_classes: int = 2,
                 video_backbone: str = "tsn_mbv2",
                 fusion_type: str = "gmu"):
        super().__init__()

        self.fusion_type = fusion_type.lower()
        self.fused_hidden = fused_hidden


        self.sensor_encoder = SensorTransformerEncoder(
            input_dim=sensor_input_dim,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            max_len=128
        )


        self.sensor_fc = nn.Linear(128, 128)
        self.sensor_bn = nn.BatchNorm1d(128)


        self.video_model = build_video_encoder(video_backbone, out_dim=video_feature_dim)
        self.video_bn = nn.BatchNorm1d(video_feature_dim)


        if self.fusion_type == "gmu":
            self.fusion = GMU2(
                sensor_dim=128,
                video_dim=video_feature_dim,
                out_dim=fused_hidden
            )
        elif self.fusion_type == "concat":
            self.fusion = ConcatFusion(
                sensor_dim=128,
                video_dim=video_feature_dim,
                out_dim=fused_hidden
            )
        elif self.fusion_type == "film":
            self.fusion = FiLMFusion(
                sensor_dim=128,
                video_dim=video_feature_dim,
                out_dim=fused_hidden
            )
        elif self.fusion_type == "xattn":
            self.fusion = CrossAttentionFusion(
                sensor_dim=128,
                video_dim=video_feature_dim,
                out_dim=fused_hidden,
                num_heads=4
            )
        elif self.fusion_type == "bilinear":
            self.fusion = BilinearFusion(
                sensor_dim=128,
                video_dim=video_feature_dim,
                out_dim=fused_hidden,
                hidden_dim=fused_hidden
            )
        else:
            raise ValueError(f"unknown fusion_type: {fusion_type}")

        self.fusion_bn = nn.BatchNorm1d(fused_hidden)
        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(fused_hidden, num_classes)


        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))


    def _fix_sensor(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        assert x.ndim == 3, f"sensor must be [B,L,N], got {tuple(x.shape)}"
        return x

    def _fix_video(self, y: torch.Tensor) -> torch.Tensor:

        if y.ndim == 6 and y.shape[1] == 1:
            y = y.squeeze(1)
        if y.ndim == 4:
            y = y.unsqueeze(1)
        assert y.ndim == 5, f"video must be [B,C,T,H,W], got {tuple(y.shape)}"
        B, C, T, H, W = y.shape
        if C == 1:
            y = y.repeat(1, 3, 1, 1, 1)
        y = (y - self.rgb_mean) / self.rgb_std
        return y

    def forward(self, x, y):

        x = self._fix_sensor(x)
        sensor_feat = self.sensor_encoder(x)
        sensor_feat = F.relu(self.sensor_bn(self.sensor_fc(sensor_feat)))


        y = self._fix_video(y)
        video_feat = self.video_model(y)
        video_feat = F.relu(self.video_bn(video_feat))


        fused = self.fusion(sensor_feat, video_feat)
        fused = F.relu(self.fusion_bn(fused))
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits
