from votenet.utils.registry import Registry

VOTE_GENERATOR_REGISTRY = Registry("VOTE_GENERATOR")
VOTE_GENERATOR_REGISTRY.__doc__ = """
Registry for vote generator, which produces point votes from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


def build_vote_generator(cfg):
    """
    Build a vote generator from `cfg.MODEL.VOTE_GENERATOR.NAME`.
    The name can be "PrecomputedProposals" to use no proposal generator.
    """
    name = cfg.MODEL.VOTE_GENERATOR.NAME

    return VOTE_GENERATOR_REGISTRY.get(name)(cfg)
