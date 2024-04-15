from utils.registry import Registry, build_from_cfg
from torch import nn

NET = Registry('net')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_net(cfg):
    return build(cfg.net, NET, default_args=dict(cfg=cfg))
