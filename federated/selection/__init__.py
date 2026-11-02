from .base import BaseClientSelector
from .all import AllSelector
from .random import RandomSelector

SELECTOR_REGISTRY = {
    "all": AllSelector,
    "random": RandomSelector,
}


def build_selector(selector_cfg: dict | None = None) -> BaseClientSelector:
    """Builds the configured client selector strategy from YAML settings."""
    selector_cfg = selector_cfg or {}
    selector_type = selector_cfg.get("type", "all")
    selector_cls = SELECTOR_REGISTRY[selector_type]
    params = {k: v for k, v in selector_cfg.items() if k != "type"}
    return selector_cls(**params)
