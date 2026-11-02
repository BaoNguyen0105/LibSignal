from abc import ABC, abstractmethod


class BaseProfile(ABC):
    """Abstract Base Class for client hardware and network constraints."""
    def __init__(self, client_id: str, buffer_size: int = 5000, compute_speed: float = 1.0):
        self.client_id = client_id
        self.buffer_size = buffer_size
        self.compute_speed = compute_speed

    @abstractmethod
    def is_dropped(self) -> bool:
        """Determines if the client's packet/weights are dropped in transit."""
        return False

    @abstractmethod
    def get_latency(self) -> float:
        """Calculates communication delay (ms) for transmitting weights."""
        return 0.0
