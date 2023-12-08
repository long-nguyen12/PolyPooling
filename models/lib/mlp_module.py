import torch
from torch import nn, Tensor
import torch.nn.functional as F
from models.models.layers import DropPath

class Pooling(nn.Module):
    def __init__(self, pool_size=3) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, 1, pool_size//2, count_include_pad=False)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x) - x


class MLPModule(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)
        # self.pooling = Pooling(pool_size=3)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PoolModule(nn.Module):
    def __init__(self, dim, pool_size=3, dpr=0., layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.token_mixer = Pooling(pool_size)
        # self.token_mixer =  nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.norm2 = nn.GroupNorm(1, dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x))) 
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(x)) 
        return x
