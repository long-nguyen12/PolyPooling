import torch
import math
from torch import nn
from models.models.backbones import *
from models.models.layers import trunc_normal_


class BaseModel(nn.Module):
    def __init__(self, backbone: str = 'Base', num_classes: int = 19) -> None:
        super().__init__()
        self.backbone = eval(backbone)('Base')
        self.backbone.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            fan_in // m.groups
            std = math.sqrt(2.0 / fan_in)
            m.weight.data.normal_(0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            print(pretrained)
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)