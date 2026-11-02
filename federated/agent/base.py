from abc import ABC, abstractmethod
from agent.rl_agent import RLAgent


class BaseFLAgent(RLAgent, ABC):
    """
    Pure Federated Traffic Signal Control Agent.
    Defines the contract for participating in federated rounds.
    """
    def __init__(self, world, rank):
        super().__init__(world, rank)

    @abstractmethod
    def get_weights(self):
        """Returns model state dict / weights to share with the server."""
        raise NotImplementedError

    @abstractmethod
    def set_weights(self, weights):
        """Updates the local model with the aggregated global weights."""
        raise NotImplementedError

    def get_sample_count(self):
        """Returns number of training samples for sample-weighted FedAvg."""
        if hasattr(self, 'replay_buffer'):
            return len(self.replay_buffer)
        return 1

    def get_client_id(self):
        """Returns the intersection ID string."""
        return str(self.id)