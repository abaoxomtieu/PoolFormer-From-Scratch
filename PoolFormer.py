from torch import nn

import torch
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import trunc_normal_


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(
        self, patch_size, stride, in_channels, embed_dim, padding=0, norm_layer=None
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):

        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(
            -1
        ).unsqueeze(-1)
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super(Pooling, self).__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False
        )

    def forward(self, x):
        return self.pool(x) - x


class PoolFormerBlock(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4, act_layer=nn.GELU, drop=0.0):
        assert (
            type(mlp_ratio) == int and mlp_ratio > 1
        ), "mlp_ratio should be an integer greater than 1"
        super(PoolFormerBlock, self).__init__()
        self.norm1 = LayerNormChannel(hidden_size)
        self.norm2 = LayerNormChannel(hidden_size)
        # self.MLP = nn.ModuleList(Mlp(hidden_size) for _ in range(4))
        self.MLP = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size) * mlp_ratio,
            act_layer=act_layer,
            drop=drop,
        )
        self.polling = Pooling()

    def forward(self, x):
        x_origin = x
        x = self.norm1(x)
        x = self.polling(x)
        x_1 = x + x_origin
        x = self.norm2(x_1)

        out = self.MLP(x)
        out = out + x_1

        return out


patch_embed_small = [[7, 4, 64], [3, 2, 128], [3, 2, 320], [3, 2, 512]]


class SmallPoolFomer(nn.Module):
    """
    A class representing a small PoolFormer model.

    This class defines the architecture of a small PoolFormer model, which consists of four stages.
    Each stage includes a PatchEmbed layer followed by multiple PoolFormerBlock layers.

    Args:
        None

    Attributes:
        stage1 (nn.Sequential): The first stage of the model.
        stage2 (nn.Sequential): The second stage of the model.
        stage3 (nn.Sequential): The third stage of the model.
        stage4 (nn.Sequential): The fourth stage of the model.

    Methods:
        forward(x): Performs forward pass through the model.

    """

    def __init__(self):
        super(SmallPoolFomer, self).__init__()
        self.stage1 = nn.Sequential(
            PatchEmbed(
                patch_embed_small[0][0],
                patch_embed_small[0][1],
                3,
                patch_embed_small[0][2],
            ),
            PoolFormerBlock(patch_embed_small[0][2]),
            PoolFormerBlock(patch_embed_small[0][2]),
        )
        self.stage2 = nn.Sequential(
            PatchEmbed(
                patch_embed_small[1][0],
                patch_embed_small[1][1],
                patch_embed_small[0][2],
                patch_embed_small[1][2],
            ),
            PoolFormerBlock(patch_embed_small[1][2]),
            PoolFormerBlock(patch_embed_small[1][2]),
        )
        self.stage3 = nn.Sequential(
            PatchEmbed(
                patch_embed_small[2][0],
                patch_embed_small[2][1],
                patch_embed_small[1][2],
                patch_embed_small[2][2],
            ),
            PoolFormerBlock(patch_embed_small[2][2]),
            PoolFormerBlock(patch_embed_small[2][2]),
            PoolFormerBlock(patch_embed_small[2][2]),
            PoolFormerBlock(patch_embed_small[2][2]),
            PoolFormerBlock(patch_embed_small[2][2]),
            PoolFormerBlock(patch_embed_small[2][2]),
        )
        self.stage4 = nn.Sequential(
            PatchEmbed(
                patch_embed_small[3][0],
                patch_embed_small[3][1],
                patch_embed_small[2][2],
                patch_embed_small[3][2],
            ),
            PoolFormerBlock(patch_embed_small[3][2]),
            PoolFormerBlock(patch_embed_small[3][2]),
        )

    def forward(self, x):
        """
        Performs forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SmallPoolFomer().to(device)
dummy = torch.rand(1, 3, 224, 224).to(device)
output = model(dummy)

from torchinfo import summary

print(summary(model, input_size=(1, 3, 256, 256)))
