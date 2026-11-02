from abc import ABC, abstractmethod


class BaseAggregator(ABC):
    """Abstract interface for all Federated Aggregation algorithms."""

    @abstractmethod
    def aggregate(self, client_weights, sample_counts=None, **kwargs):
        """
        Takes:
          - client_weights: list of state_dict dictionaries from participating clients
          - sample_counts: optional list of sample counts (for weighted FedAvg)
        Returns:
          - global_weights: a single aggregated state_dict dictionary
        """
        raise NotImplementedError