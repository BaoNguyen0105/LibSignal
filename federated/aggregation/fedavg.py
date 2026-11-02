import torch
from federated.aggregation.base import BaseAggregator


class FedAvgAggregator(BaseAggregator):
    """
    Standard Federated Averaging (FedAvg).
    Supports both equal-weight averaging and sample-weighted averaging.
    """
    def __init__(self, weighted=False):
        self.weighted = weighted

    def aggregate(self, client_weights, sample_counts=None, **kwargs):
        if not client_weights:
            return {}

        # 1. Grab all parameter keys (e.g. 'dense_1.weight', 'dense_1.bias'...)
        param_keys = client_weights[0].keys()
        global_weights = {}

        # 2. Case A: Sample-Weighted FedAvg (Busy intersections get more vote)
        if self.weighted and sample_counts is not None and sum(sample_counts) > 0:
            total_samples = sum(sample_counts)
            coefficients = [n / total_samples for n in sample_counts]
            
            for key in param_keys:
                # Sum of (w_k * coeff_k)
                global_weights[key] = sum(
                    cw[key].float() * coeff 
                    for cw, coeff in zip(client_weights, coefficients)
                )

        # 3. Case B: Standard Equal FedAvg (Simple average across all clients)
        else:
            for key in param_keys:
                # Stack all client tensors along a new dimension and take the mean
                global_weights[key] = torch.stack(
                    [cw[key].float() for cw in client_weights]
                ).mean(dim=0)

        return global_weights