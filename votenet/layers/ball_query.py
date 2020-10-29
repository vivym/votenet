import torch
from torch.autograd import Function

from votenet import _C

__all__ = ["ball_query"]


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, num_samples, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        num_samples : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indices of the features that form the query balls
        """
        inds = _C.ball_query(new_xyz, xyz, radius, num_samples)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply
