from .base import BaseProfile
from .simple import SimpleProfile

PROFILE_REGISTRY = {
    "simple": SimpleProfile,
}

def build_profile(client_id: str, profile_cfg: dict) -> BaseProfile:
    """Builds the configured profile subtype based on YAML configuration."""
    profile_type = profile_cfg.get("type", "simple")
    profile_cls = PROFILE_REGISTRY.get(profile_type, SimpleProfile)
    return profile_cls.from_config(client_id, profile_cfg)