from abc import ABC, abstractmethod
from collections import deque


class BaseFLAgent(ABC):
    """
    Pure Federated Traffic Signal Control Agent Mixin.
    Defines the contract for participating in federated rounds.
    """
    def __init__(self):
        self.round_sample_count: int = 0

    @abstractmethod
    def get_weights(self):
        """Returns model state dict / weights to share with the server."""
        raise NotImplementedError

    @abstractmethod
    def set_weights(self, weights):
        """Updates the local model with the aggregated global weights."""
        raise NotImplementedError

    def set_buffer_size(self, buffer_size: int):
        """Resizes the agent's local replay buffer according to its hardware profile."""
        self.buffer_size = buffer_size
        self.replay_buffer = deque(self.replay_buffer, maxlen=buffer_size)

    @abstractmethod
    def get_id(self):
        """Returns the intersection ID string."""
        raise NotImplementedError

    def get_sample_count(self):
        """Returns number of training samples/vehicles for sample-weighted FedAvg."""
        return max(1, self.round_sample_count)

    def reset_sample_count(self):
        """Resets the per-round sample counter after a federated round."""
        self.round_sample_count = 0
