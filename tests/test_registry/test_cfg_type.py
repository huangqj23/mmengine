# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch.nn as nn

from mmengine.config import Config
from mmengine.config.lazy import LazyObject
from mmengine.model.weight_init import PretrainedInit
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm
from mmengine.registry import (MODELS, Registry, cfg_type_matches,
                               cfg_type_name, registered_names,
                               resolve_cfg_type)


@pytest.fixture(autouse=True)
def _register_bricks():
    # BN / SyncBN / GN / nearest / bilinear aliases live in mmcv.cnn
    pytest.importorskip('mmcv.cnn')


def test_string_type_semantics_unchanged():
    assert cfg_type_matches('BN', 'BN')
    assert not cfg_type_matches('BN', 'GN')
    assert not cfg_type_matches('BN', 'BN2d')  # plain-name compare stays exact
    assert cfg_type_matches('Pretrained', 'Pretrained')
    assert cfg_type_matches(None, None)
    assert not cfg_type_matches(None, 'BN')


def test_class_matches_registered_aliases():
    assert cfg_type_matches(nn.BatchNorm2d, 'BN')
    assert cfg_type_matches(nn.BatchNorm2d, 'BN2d')
    # 'SyncBN' is registered as mmengine's SyncBatchNorm wrapper, not the
    # torch class; matching is by identity, so only the wrapper matches.
    assert cfg_type_matches(SyncBatchNorm, 'SyncBN')
    assert cfg_type_matches(nn.GroupNorm, 'GN')
    assert cfg_type_matches(nn.Upsample, 'nearest', 'bilinear')
    assert cfg_type_matches(PretrainedInit, 'Pretrained')
    assert not cfg_type_matches(nn.BatchNorm2d, 'GN')
    # class object itself in the candidates (mmdet ``(str, cls)`` pattern)
    assert cfg_type_matches(nn.ReLU, ('ReLU', nn.ReLU)[1])
    assert cfg_type_matches(nn.ReLU, 'ReLU', nn.LeakyReLU)
    # __name__ fallback for unregistered classes
    class Foo:
        pass
    assert cfg_type_matches(Foo, 'Foo')
    assert not cfg_type_matches(Foo, 'Bar')


def test_dotted_strings_resolve():
    # full module path (what Config.dump writes for a class type) resolves
    assert cfg_type_matches('torch.nn.modules.batchnorm.BatchNorm2d', 'BN')
    assert not cfg_type_matches('no.such.Module', 'BN')
    # scope-prefixed names keep plain string semantics
    assert not cfg_type_matches('mmdet.Mosaic', 'Mosaic')
    assert cfg_type_matches('mmdet.Mosaic', 'mmdet.Mosaic')


def test_lazy_object():
    lazy = LazyObject('torch.nn', 'BatchNorm2d')
    assert cfg_type_matches(lazy, 'BN')
    assert resolve_cfg_type(lazy) is nn.BatchNorm2d


def test_registered_names_and_name():
    assert registered_names(nn.BatchNorm2d)[:2] == ('BN', 'BN2d')
    assert registered_names(object()) == ()
    assert cfg_type_name(nn.BatchNorm2d) == 'BN'
    assert cfg_type_name('BN') == 'BN'
    assert cfg_type_name(PretrainedInit) == 'Pretrained'

    class Unregistered:
        pass
    assert cfg_type_name(Unregistered) == 'Unregistered'


def test_index_refreshes_after_registration():
    reg = Registry('cfg_type_test')

    class Later:
        pass
    assert registered_names(Later) == ()
    reg.register_module('later_alias', module=Later)
    assert 'later_alias' in registered_names(Later)
    assert cfg_type_matches(Later, 'later_alias')


def test_resolve_cfg_type():
    assert resolve_cfg_type('BN') is nn.BatchNorm2d
    assert resolve_cfg_type(nn.BatchNorm2d) is nn.BatchNorm2d
    with pytest.raises(KeyError):
        resolve_cfg_type('NoSuchNormXYZ')
    with pytest.raises(TypeError):
        resolve_cfg_type(123)


def test_registry_get_accepts_class():
    assert MODELS.get(nn.ReLU) is nn.ReLU
    assert MODELS.get(LazyObject('torch.nn', 'ReLU')) is nn.ReLU
    assert MODELS.get('ReLU') is nn.ReLU
    with pytest.raises(TypeError):
        MODELS.get(123)


def test_pretty_text_with_real_class():
    cfg = Config(dict(a=dict(type=nn.ReLU, inplace=True)))
    text = cfg.pretty_text
    assert "'torch.nn.modules.activation.ReLU'" in text
    assert '<class' not in text
    assert cfg.to_dict()['a']['type'] == 'torch.nn.modules.activation.ReLU'
