from abc import ABC, abstractmethod


class BaseFLMetric(ABC):
    """Abstract interface for individual Federated & Traffic evaluation metrics."""
    name: str = "base_metric"
    scope: str = "round"  # "episode" or "round"

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def compute(self, **context) -> float:
        """Computes the scalar metric value from the provided episode or round context."""
        raise NotImplementedError

    def reset(self) -> None:
        """Resets any internal state across runs."""
        pass

