from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml
from .hamburger import HamBurger
from ..bricks import SeprableConv2d, ConvRelu, ConvBNRelu, resize

with open("./models/models/config.yaml") as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)


class HamDecoder(nn.Module):
    def __init__(self, outChannels, enc_embed_dims=[64, 128, 320, 512]):
        super().__init__()
        enc_embed_dims = [64, 128, 320, 512]
        # enc_embed_dims = [96, 192, 384, 768]
        print(f"Ham Decoder {enc_embed_dims}")
        ham_channels = config["ham_channels"]
        self.squeeze = ConvRelu(sum(enc_embed_dims), ham_channels)
        self.ham_attn = HamBurger(ham_channels, config)
        self.align = ConvRelu(ham_channels, 1)

    def forward(self, features):
        features = [
            resize(feature, size=features[0].shape[2:], mode="bicubic")
            for feature in features
        ]
        x = torch.cat(features, dim=1)

        x = self.squeeze(x)
        x = self.ham_attn(x)
        x = self.align(x)

        return x
