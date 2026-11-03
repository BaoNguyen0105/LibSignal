from federated.metrics.base import BaseFLMetric


class DelayMetric(BaseFLMetric):
    """
    LibSignal Original Metric: Mean Vehicle Delay ("delay").
    Average approximate or real delay across lanes over the episode.
    """
    name = "delay"
    scope = "episode"

    def compute(self, **context) -> float:
        metric = context.get("metric")
        if metric is None:
            return 0.0
        val = metric.delay()
        return float(val) if val is not None else 0.0

