import math

import torch
import torch.nn as nn


def build_positional_encoding_layer(type, dim, fuse_type="cat", with_extra_identity=False):
    return PositionalEncoding(type, dim, fuse_type, with_extra_identity)


class PositionalEncoding(nn.Module):
    def __init__(self, type, dim, fuse_type, with_extra_identity):
        super().__init__()

        self.type = type
        self.dim = dim
        self.fuse_type = fuse_type
        self.with_extra_identity = with_extra_identity

        # TODO:
        assert type == "Identity"
        if type == "Identity":
            self.num_extra_channels = 3
        elif fuse_type == "addition":
            self.num_extra_channels = 0
        else:
            self.num_extra_channels = dim

        if type == "Gaussian":
            m = dim - 2 if with_extra_identity else dim
            self.register_buffer("b", torch.randn(1, m // 2, 2))

    def forward(self, coords, features, norm_term=None):
        batch_size = features.size(0)
        identity = coords

        if self.type == "Identity":
            # do nothing
            pass
        elif self.type == "Transformer":
            pass
        elif self.type == "NeRF":
            dim = self.dim - 2 if self.with_extra_identity else self.dim
            L = dim // 4
            mul_term = math.pi * 2 ** torch.arange(
                0, L, 1, dtype=features.dtype
            ).to(device=features.device).reshape(1, L, 1).expand(batch_size, -1, -1)

            x = coords[:, 0, :].reshape(batch_size, 1, -1).repeat(1, dim // 2, 1)
            x = torch.cat([
                torch.sin(x[:, :L, :] * mul_term),
                torch.cos(x[:, L:, :] * mul_term),
            ], dim=1)

            y = coords[:, 1, :].reshape(batch_size, 1, -1).repeat(1, dim // 2, 1)
            y = torch.cat([
                torch.sin(y[:, :L, :] * mul_term),
                torch.cos(y[:, L:, :] * mul_term),
            ], dim=1)

            coords = torch.cat([x, y], dim=1)
        elif self.type == "Gaussian":
            dim = self.dim - 2 if self.with_extra_identity else self.dim
            coords = coords.permute(0, 2, 1).reshape(-1, 2, 1) * 2 * math.pi
            coords = torch.bmm(
                self.b.expand(coords.size(0), -1, -1), coords
            ).reshape(batch_size, -1, dim // 2).permute(0, 2, 1)
            coords = torch.cat([coords.cos(), coords.sin()], dim=1)
        elif self.type == "Repeat":
            coords = coords.repeat(1, self.dim // 2, 1)
        else:
            raise RuntimeError("invalid positional encoding type")

        if self.with_extra_identity:
            coords = torch.cat([identity, coords], dim=1)

        if norm_term is not None:
            coords = coords * norm_term

        if self.fuse_type == "addition":
            return coords + features
        else:
            return torch.cat([coords, features], dim=1)
