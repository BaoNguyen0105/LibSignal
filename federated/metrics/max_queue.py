import numpy as np
from federated.metrics.base import BaseFLMetric


class MaxQueueMetric(BaseFLMetric):
    """
    Worst-Intersection Queue Length ("max_queue").
    Measures the maximum average queue length among all intersections in the road network
    to expose localized gridlock that city-wide averages hide.
    """
    name = "max_queue"
    scope = "episode"

    def compute(self, **context) -> float:
        metric = context.get("metric")
        if metric is None:
            return 0.0
        lane_queues = metric.lane_queue()
        if lane_queues is None or len(lane_queues) == 0:
            return 0.0
        return float(np.max(lane_queues))

