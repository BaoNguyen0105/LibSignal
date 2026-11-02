from abc import ABC, abstractmethod


class BaseProfile(ABC):
    """Abstract Base Class for client hardware and network constraints."""
    def __init__(self, client_id: str, buffer_size: int = 5000):
        self.client_id = client_id
        self.buffer_size = buffer_size

    @abstractmethod
    def is_dropped(self) -> bool:
        """Determines if the client's packet/weights are dropped in transit."""
        raise NotImplementedError

    @abstractmethod
    def calculate_upload_delay_ms(self, model_size_mb: float) -> float:
        """Calculates communication delay for transmitting weights."""
        raise NotImplementedError