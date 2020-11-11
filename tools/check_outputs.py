import torch


def check_seeds():
    legacy_seeds = torch.load("../_votenet/seeds.pth", map_location="cpu")
    seeds = torch.load("seeds.pth", map_location="cpu")

    for k in ["seed_xyz", "seed_features", "seed_inds"]:
        assert (legacy_seeds[k] == seeds[k]).all()

    print("done")


def check_votes():
    legacy_m = torch.load("../_votenet/votes.pth", map_location="cpu")
    m = torch.load("votes.pth", map_location="cpu")

    for k in ["voted_xyz", "voted_features", "voted_inds"]:
        assert (legacy_m[k] == m[k]).all(), k
        print(k, "checked.")

    print("done")


def main():
    legacy_m = torch.load("../_votenet/rpn_head.pth", map_location="cpu")
    m = torch.load("rpn_head.pth", map_location="cpu")

    legacy_m = legacy_m.permute(0, 2, 1)
    print(legacy_m.size(), m.size())

    assert (legacy_m == m).all()

    print("done")


if __name__ == '__main__':
    main()
