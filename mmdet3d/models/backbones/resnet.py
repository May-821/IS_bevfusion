from typing import List, Tuple

import torch
from mmcv.cnn.resnet import BasicBlock, make_res_layer
from mmcv.cnn import build_conv_layer, build_norm_layer
from torch import nn

from mmdet.models import BACKBONES

__all__ = ["GeneralizedResNet", "GeneralizedResNetV2"]


@BACKBONES.register_module()
class GeneralizedResNet(nn.ModuleList):
    def __init__(
        self,
        in_channels: int,
        blocks: List[Tuple[int, int, int]],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.blocks = blocks

        for num_blocks, out_channels, stride in self.blocks:
            blocks = make_res_layer(
                BasicBlock,
                in_channels,
                out_channels,
                num_blocks,
                stride=stride,
                dilation=1,
            )
            in_channels = out_channels
            self.append(blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for module in self:
            x = module(x)
            outputs.append(x)
        return outputs
    
@BACKBONES.register_module()
class GeneralizedResNetV2(nn.ModuleList):
    def __init__(
        self,
        in_channels: int,
        blocks: List[Tuple[int, int, int]],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.blocks = blocks

        for num_blocks, out_channels, stride in self.blocks:
            blocks = make_res_layer(
                BasicBlock,
                in_channels,
                out_channels,
                num_blocks,
                stride=stride,
                dilation=1,
            )
            in_channels = out_channels
            self.append(blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # If x is a list, use the first tensor (or handle this case as needed)
        if isinstance(x, list):
            x = x[0]  # Assuming you want to process the first element of the list
        
        # Ensure the input tensor is of type FloatTensor
        x = x.to(torch.float32)  # Convert x to FloatTensor (32-bit)
        
        outputs = []
        for module in self:
            x = module(x)
            outputs.append(x)
        return outputs