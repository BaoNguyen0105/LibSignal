from .base import BaseProfile
from .simple import SimpleProfile

PROFILE_REGISTRY = {
    "simple": SimpleProfile,
}


def build_profile(client_id: str, profile_cfg: dict | None = None) -> BaseProfile:
    """Builds the configured profile subtype based on YAML configuration."""
    profile_cfg = profile_cfg or {}
    profile_type = profile_cfg.get("type", "simple")
    profile_cls = PROFILE_REGISTRY[profile_type]
    return profile_cls.from_config(client_id, profile_cfg)
