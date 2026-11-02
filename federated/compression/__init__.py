from .base import BaseCompressor
from .none import NoCompression

COMPRESSOR_REGISTRY = {
    "none": NoCompression,
}


def build_compressor(compressor_cfg: dict | None = None) -> BaseCompressor:
    """Builds the configured weight compressor strategy from YAML settings."""
    compressor_cfg = compressor_cfg or {}
    comp_type = compressor_cfg.get("type", "none")
    comp_cls = COMPRESSOR_REGISTRY[comp_type]
    params = {k: v for k, v in compressor_cfg.items() if k != "type"}
    return comp_cls(**params)
