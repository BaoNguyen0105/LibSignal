import numpy as np
from federated.metrics.base import BaseFLMetric


class QueueFairnessMetric(BaseFLMetric):
    """
    Jain's Fairness Index over Intersection Queues ("queue_fairness").
        J(q) = (sum_j q_j)^2 / (N * sum_j q_j^2) in [1/N, 1.0]
    A value near 1.0 indicates balanced traffic queues across all intersections;
    a low value indicates severe spatial unfairness (some intersections are starved).
    """
    name = "queue_fairness"
    scope = "episode"

    def compute(self, **context) -> float:
        metric = context.get("metric")
        if metric is None:
            return 1.0
        lane_queues = metric.lane_queue()
        if lane_queues is None or len(lane_queues) == 0:
            return 1.0

        q = np.asarray(lane_queues, dtype=np.float64).flatten()
        n = len(q)
        denom = float(n * np.sum(q ** 2))
        if denom <= 1e-12:
            return 1.0
        num = float(np.sum(q) ** 2)
        return num / denom

