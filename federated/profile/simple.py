import random
from federated.profile.base import BaseProfile


class SimpleProfile(BaseProfile):
    """Models an edge computing controller (e.g. at a traffic intersection)."""
    def __init__(
        self,
        client_id: str,
        buffer_size: int = 5000,
        compute_speed: float = 1.0,
        drop_rate: float = 0.0,
        latency_ms: float = 20.0,
        bandwidth_mbps: float = 10.0,
        model_size_mb: float = 0.5,
    ):
        super().__init__(
            client_id=client_id,
            buffer_size=buffer_size,
            compute_speed=compute_speed,
        )
        self.drop_rate = drop_rate
        self.latency_ms = latency_ms
        self.bandwidth_mbps = bandwidth_mbps
        self.model_size_mb = model_size_mb

    @classmethod
    def from_config(cls, client_id: str, profile_cfg: dict) -> "SimpleProfile":
        """Builds a SimpleProfile from YAML default + override settings."""
        params = profile_cfg.get("default", {}).copy()
        overrides = profile_cfg.get("overrides", {}).get(client_id, {})
        params.update(overrides)
        return cls(client_id=client_id, **params)

    def is_dropped(self) -> bool:
        return random.random() < self.drop_rate

    def get_latency(self) -> float:
        transmission_time_ms = ((self.model_size_mb * 8) / self.bandwidth_mbps) * 1000.0
        return self.latency_ms + transmission_time_ms
