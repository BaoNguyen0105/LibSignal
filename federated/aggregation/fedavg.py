import torch
from federated.aggregation.base import BaseAggregator


class FedAvgAggregator(BaseAggregator):
    """
    Standard Federated Averaging (FedAvg).
    Supports both equal-weight averaging and sample-weighted averaging.
    """
    def __init__(self, weighted: bool = False, **kwargs):
        self.weighted = weighted

    def aggregate(self, client_weights, sample_counts=None, **kwargs):
        param_keys = client_weights[0].keys()
        global_weights = {}

        if self.weighted:
            total_samples = sum(sample_counts)
            coefficients = [n / total_samples for n in sample_counts]
            for key in param_keys:
                global_weights[key] = sum(
                    cw[key].float() * coeff
                    for cw, coeff in zip(client_weights, coefficients)
                )
        else:
            for key in param_keys:
                global_weights[key] = torch.stack(
                    [cw[key].float() for cw in client_weights]
                ).mean(dim=0)

        return global_weights
