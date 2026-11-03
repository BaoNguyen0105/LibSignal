from federated.metrics.base import BaseFLMetric


class RewardsMetric(BaseFLMetric):
    """
    LibSignal Original Metric: Mean Episode Reward ("rewards").
    Average step reward across intersections over the episode.
    """
    name = "rewards"
    scope = "episode"

    def compute(self, **context) -> float:
        metric = context.get("metric")
        if metric is None:
            return 0.0
        return float(metric.rewards())

