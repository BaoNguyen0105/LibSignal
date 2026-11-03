import math
import torch
from federated.metrics.base import BaseFLMetric


class WeightDriftMetric(BaseFLMetric):
    """
    Client-to-Global Weight Divergence ("weight_drift").
    Computes the mean L2 distance ||w_i - w_global||_2 across active clients
    prior to aggregation to quantify non-IID local policy drift.
    """
    name = "weight_drift"
    scope = "round"

    def compute(self, **context) -> float:
        client_weights = context.get("client_weights") or []
        global_weights = context.get("global_weights")
        if not client_weights or global_weights is None:
            return 0.0

        drifts = []
        for cw in client_weights:
            sq_sum = 0.0
            for key, g_tensor in global_weights.items():
                if key not in cw or not torch.is_floating_point(g_tensor):
                    continue
                diff = cw[key].detach().float() - g_tensor.detach().float()
                sq_sum += float(torch.sum(diff * diff).item())
            drifts.append(math.sqrt(max(sq_sum, 0.0)))

        return float(sum(drifts) / len(drifts)) if drifts else 0.0

