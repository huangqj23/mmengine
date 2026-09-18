# Copyright (c) OpenMMLab. All rights reserved.
"""Helpers that make a config ``type`` written as a class/function behave
exactly like the same ``type`` written as a registered-name string.

Pure-Python style configs may write ``type=ResNet`` instead of
``type='ResNet'``.  ``build_from_cfg`` accepts both, but a lot of code reads
``cfg['type']`` *before* building and compares it with string literals
(``norm_cfg['type'] == 'BN'``, ``loss['type'] in ['FocalLoss']`` ...).  Those
comparisons silently take the wrong branch for a class.  Use these helpers
instead of raw comparisons::

    from mmengine.registry import cfg_type_matches
    if cfg_type_matches(norm_cfg['type'], 'BN', 'SyncBN'):
        ...

``cfg_type_matches(nn.BatchNorm2d, 'BN')`` is ``True`` because the alias is
looked up in the registries (``nn.BatchNorm2d`` is registered as both ``'BN'``
and ``'BN2d'``), so aliases that were converted to classes keep behaving as
before.
"""
import inspect
from typing import Any, Dict, Optional, Tuple

__all__ = [
    'cfg_type_matches', 'cfg_type_name', 'resolve_cfg_type',
    'registered_names'
]

# id(obj) -> (obj, names).  ``obj`` is kept so that the id cannot be reused.
# The index only reflects what is registered at call time; it never imports
# anything itself (a comparison must not mutate registries as a side effect).
# That is sufficient: aliases are registered by the modules that define or
# wrap the class, and code comparing against an alias ('BN', 'Pretrained',
# ...) lives in, or imports, those modules.
_INDEX: Dict[int, Tuple[Any, Tuple[str, ...]]] = {}
_INDEX_VERSION = -1
_STR_CACHE: Dict[str, Any] = {}


def _build_lazy(t: Any) -> Any:
    from mmengine.config.lazy import LazyAttr, LazyObject
    if isinstance(t, (LazyObject, LazyAttr)):
        return t.build()
    return t


def _all_registries():
    from .registry import Registry
    regs = list(Registry._instances)
    roots = sorted((r for r in regs if r.parent is None), key=lambda r: r.name)
    children = sorted((r for r in regs if r.parent is not None),
                      key=lambda r: (r.scope or '', r.name))
    return roots + children


def _rebuild_index() -> None:
    global _INDEX, _INDEX_VERSION
    from .registry import Registry
    idx: Dict[int, Tuple[Any, list]] = {}
    for reg in _all_registries():
        for name, obj in reg._module_dict.items():
            entry = idx.setdefault(id(obj), (obj, []))
            entry[1].append(name)
            if reg.parent is not None and reg.scope:
                entry[1].append(f'{reg.scope}.{name}')
    _INDEX = {k: (o, tuple(n)) for k, (o, n) in idx.items()}
    _INDEX_VERSION = Registry._version


def registered_names(obj: Any) -> Tuple[str, ...]:
    """All names ``obj`` is registered under, in registration order.

    Root registries come first (sorted by registry name), then child
    registries (sorted by scope).  For a child registry both ``'name'`` and
    ``'scope.name'`` are listed.  Returns ``()`` for unregistered objects.
    """
    from .registry import Registry
    if _INDEX_VERSION != Registry._version:
        _rebuild_index()
    entry = _INDEX.get(id(obj))
    if entry is None or entry[0] is not obj:
        return ()
    return entry[1]


def _is_module_path(s: str) -> bool:
    """``'pkg.mod.Name'`` (>= 2 dots), as produced by ``Config.dump`` for a
    class ``type``.  ``'scope.Name'`` is *not* a module path: scope-prefixed
    registered names keep plain string semantics."""
    return s.count('.') >= 2


def _resolve_module_path(s: str) -> Optional[Any]:
    """``'torch.nn.modules.batchnorm.BatchNorm2d'`` -> object; None if the
    path cannot be imported."""
    if s in _STR_CACHE:
        return _STR_CACHE[s]
    try:
        from mmengine.utils import get_object_from_string
        obj = get_object_from_string(s)
    except Exception:  # noqa: BLE001
        obj = None
    _STR_CACHE[s] = obj
    return obj


def _obj_matches(t: Any, names: tuple) -> bool:
    if any(t is n for n in names):
        return True
    name = getattr(t, '__name__', None)
    if name is not None and name in names:
        return True
    str_names = {n for n in names if isinstance(n, str)}
    if not str_names:
        return False
    return bool(set(registered_names(t)) & str_names)


def cfg_type_matches(t: Any, *names: Any) -> bool:
    """Whether config ``type`` value ``t`` denotes any of ``names``.

    ``names`` may mix registered-name strings and class/function objects.

    * ``t`` is a string: ``True`` if it equals one of the names (unchanged
      semantics; ``'mmdet.Mosaic'`` does not match ``'Mosaic'``).  A full
      module path such as ``'torch.nn.modules.batchnorm.BatchNorm2d'`` (what
      ``Config.dump`` writes for a class ``type``) is additionally resolved
      to the object and compared like a class.
    * ``t`` is a class/function (or a ``LazyObject``): ``True`` if it *is* one
      of the names, its ``__name__`` is one of the names, or it is registered
      under one of the names in any registry.
    * ``t`` is ``None``: ``True`` only if ``None`` is among the names.
    """
    t = _build_lazy(t)
    if t is None:
        return any(n is None for n in names)
    if isinstance(t, str):
        if t in names:
            return True
        if _is_module_path(t):
            obj = _resolve_module_path(t)
            if obj is not None:
                return _obj_matches(obj, names)
        return False
    return _obj_matches(t, names)


def cfg_type_name(t: Any) -> str:
    """A stable string name for logging / dict keys.

    Strings are returned unchanged.  For a class the first registered name in
    the current default scope is preferred, then the first root-registry
    name, then ``__name__``.
    """
    t = _build_lazy(t)
    if isinstance(t, str):
        return t
    if t is None:
        return 'None'
    names = registered_names(t)
    if names:
        from .default_scope import DefaultScope
        inst = DefaultScope.get_current_instance()
        scope = inst.scope_name if inst is not None else None
        if scope:
            for n in names:
                if n.startswith(scope + '.'):
                    return n[len(scope) + 1:]
        for n in names:
            if '.' not in n:
                return n
        return names[0]
    return getattr(t, '__name__', None) or str(t)


def resolve_cfg_type(t: Any, registry: Any = None) -> Any:
    """Config ``type`` value -> class/function.

    Strings are looked up in ``registry`` (``mmengine.registry.MODELS`` by
    default, searching from the current scope up to the root); classes,
    functions and ``LazyObject`` are returned as the object itself.
    """
    t = _build_lazy(t)
    if isinstance(t, str):
        if registry is None:
            from .root import MODELS
            registry = MODELS
        with registry.switch_scope_and_registry(None) as reg:
            obj = reg.get(t)
        if obj is None:
            raise KeyError(f'Cannot find {t} in registry under scope name '
                           f'{registry.scope}')
        return obj
    if inspect.isclass(t) or callable(t):
        return t
    raise TypeError(f'type must be a str, a class or a callable, got {t!r}')
