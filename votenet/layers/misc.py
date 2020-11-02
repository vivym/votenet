import torch


def nn_distance(pc1, pc2, dist="l1"):
    """
    Input:
        pc1: (B, N, C) / (N, C) torch tensor
        pc2: (B, M, C) / (M, C) torch tensor
    Output:
        dist1: (B, N) torch float32 tensor
        idx1: (B, N) torch int64 tensor
        dist2: (B, M) torch float32 tensor
        idx2: (B, M) torch int64 tensor
    """
    assert pc1.dim() == pc2.dim()
    if pc1.dim() == 2:
        need_squeeze = True
        pc1 = pc1.unsqueeze(0)
        pc2 = pc2.unsqueeze(0)
    else:
        need_squeeze = False

    n = pc1.shape[1]
    m = pc2.shape[1]

    pc1 = pc1.unsqueeze(2).repeat(1, 1, m, 1)
    pc2 = pc2.unsqueeze(1).repeat(1, n, 1, 1)
    pc_diff = pc1 - pc2

    if dist == "l1":
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)  # b, n, m
    elif dist == "euclidean":
        pc_dist = torch.sum(pc_diff ** 2, dim=-1)
    else:
        raise NotImplementedError

    dist1, idx1 = torch.min(pc_dist, dim=2)  # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1)  # (B,M)

    if need_squeeze:
        dist1.squeeze_(0)
        idx1.squeeze_(0)
        dist2.squeeze_(0)
        idx2.squeeze_(0)

    return dist1, idx1, dist2, idx2
