from abc import ABC, abstractmethod


class BaseProfile(ABC):
    """Abstract Base Class for client hardware and network constraints."""
    def __init__(self, client_id: str, buffer_size: int = 5000, compute_speed: float = 1.0):
        self.client_id = client_id
        self.buffer_size = buffer_size
        self.compute_speed = compute_speed

    def step_state(self) -> None:
        """Advances any dynamic channel, congestion, or hardware state for a new round."""
        pass

    def get_compute_speed(self) -> float:
        """Returns the relative hardware compute speed multiplier (> 0)."""
        return max(float(self.compute_speed), 1e-6)

    def get_payload_size_mb(self, compression_ratio: float = 1.0) -> float:
        """Returns the transmitted model payload size in MB after compression."""
        base_mb = float(getattr(self, "model_size_mb", 0.5))
        return max(0.0, base_mb * max(float(compression_ratio), 1e-4))

    @abstractmethod
    def get_drop_rate(self) -> float:
        """Returns the packet drop probability in [0.0, 1.0]."""
        raise NotImplementedError

    @abstractmethod
    def is_dropped(self) -> bool:
        """Determines if the client's packet/weights are dropped in transit."""
        raise NotImplementedError

    @abstractmethod
    def get_latency(self, compression_ratio: float = 1.0) -> float:
        """Calculates communication delay (ms) for transmitting weights."""
        raise NotImplementedError

    def get_round_time(
        self,
        base_compute_ms: float = 100.0,
        compression_ratio: float = 1.0,
    ) -> float:
        """Estimates total round completion time (compute + communication in ms)."""
        compute_time_ms = float(base_compute_ms) / self.get_compute_speed()
        comm_time_ms = max(float(self.get_latency(compression_ratio=compression_ratio)), 0.0)
        return max(compute_time_ms + comm_time_ms, 1e-6)
