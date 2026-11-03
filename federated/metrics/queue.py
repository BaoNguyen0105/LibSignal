from federated.metrics.base import BaseFLMetric


class QueueMetric(BaseFLMetric):
    """
    LibSignal Original Metric: Mean Queue Length ("queue").
    Average number of waiting vehicles across all intersections and steps.
    """
    name = "queue"
    scope = "episode"

    def compute(self, **context) -> float:
        metric = context.get("metric")
        if metric is None:
            return 0.0
        val = metric.queue()
        return float(val) if val is not None else 0.0

