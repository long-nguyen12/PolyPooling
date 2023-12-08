import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.lib.context_module import CFPModule
from models.lib.mlp_module import PoolModule
from models.models.base import BaseModel
from models.models.heads import HamDecoder
from models.lib.cbam import CBAM


class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class PoolPolyp(BaseModel):
    def __init__(
        self, backbone: str = "PoolFormer", num_classes: int = 1, channels=320
    ) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = HamDecoder(outChannels=num_classes)
        self.apply(self._init_weights)

        self.ra4_conv1 = BasicConv2d(512, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(320, 160, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(160, 160, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(160, 160, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(160, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(128, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 1 ----
        self.ra1_conv1 = BasicConv2d(64, 64, kernel_size=1)
        self.ra1_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra1_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra1_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.cbam1 = CBAM(gate_channels=64)
        self.cbam2 = CBAM(gate_channels=128)
        self.cbam3 = CBAM(gate_channels=320)
        self.cbam4 = CBAM(gate_channels=512)

        self.CFP_1 = CFPModule(64, d=8)
        self.CFP_2 = CFPModule(128, d=8)
        self.CFP_3 = CFPModule(320, d=8)
        self.CFP_4 = CFPModule(512, d=8)

        self.MLP_1 = PoolModule(64)
        self.MLP_2 = PoolModule(128)
        self.MLP_3 = PoolModule(320)
        self.MLP_4 = PoolModule(512)

    def forward(self, x: Tensor) -> Tensor:
        x_size = x.size()[2:]

        y = self.backbone(x)
        # apply RA module for each feature block
        x1, x2, x3, x4 = y

        # apply CBAM module for each feature output block
        y1 = self.cbam1(x1)
        y2 = self.cbam2(x2)
        y3 = self.cbam3(x3)
        y4 = self.cbam4(x4)

        x1_size = x1.size()[2:]
        x2_size = x2.size()[2:]
        x3_size = x3.size()[2:]
        x4_size = x4.size()[2:]

        # decode head
        y = self.decode_head([y1, y2, y3, y4])  # 4x reduction in image size
        y = F.interpolate(
            y, size=x.shape[2:], mode="bicubic", align_corners=True
        )  # to original image shape

        # compute
        y5_4 = F.interpolate(y, size=x4_size, mode="bicubic", align_corners=True)
        x_cfp_4 = self.CFP_4(x4)
        x = -1 * (torch.sigmoid(y5_4)) + 1
        x_mlp_4 = self.MLP_4(x_cfp_4)
        x = x.expand(-1, 512, -1, -1).mul(x_mlp_4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + y5_4
        score4 = F.interpolate(x, x_size, mode="bicubic", align_corners=True)

        y4_3 = F.interpolate(x, x3_size, mode="bicubic", align_corners=True)
        x_cfp_3 = self.CFP_3(x3)
        x = -1 * (torch.sigmoid(y4_3)) + 1
        x_mlp_3 = self.MLP_3(x_cfp_3)
        x = x.expand(-1, 320, -1, -1).mul(x_mlp_3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + y4_3
        score3 = F.interpolate(x, x_size, mode="bicubic", align_corners=True)

        y3_2 = F.interpolate(x, x2_size, mode="bicubic", align_corners=True)
        x_cfp_2 = self.CFP_2(x2)
        x = -1 * (torch.sigmoid(y3_2)) + 1
        x_mlp_2 = self.MLP_2(x_cfp_2)
        x = x.expand(-1, 128, -1, -1).mul(x_mlp_2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + y3_2
        score2 = F.interpolate(x, x_size, mode="bicubic", align_corners=True)

        y2_1 = F.interpolate(x, x1_size, mode="bicubic", align_corners=True)
        x_cfp_1 = self.CFP_1(x1)
        x = -1 * (torch.sigmoid(y2_1)) + 1
        x_mlp_1 = self.MLP_1(x_cfp_1)
        x = x.expand(-1, 64, -1, -1).mul(x_mlp_1)
        x = self.ra1_conv1(x)
        x = F.relu(self.ra1_conv2(x))
        x = F.relu(self.ra1_conv3(x))
        ra1_feat = self.ra1_conv4(x)
        x = ra1_feat + y2_1
        score1 = F.interpolate(x, x_size, mode="bicubic", align_corners=True)

        return y, score4, score3, score2, score1
