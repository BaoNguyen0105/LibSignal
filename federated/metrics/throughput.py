from federated.metrics.base import BaseFLMetric


class ThroughputMetric(BaseFLMetric):
    """
    LibSignal Original Metric: Network Throughput ("throughput").
    Total number of vehicles that have completed their trips by the end of the episode.
    """
    name = "throughput"
    scope = "episode"

    def compute(self, **context) -> float:
        metric = context.get("metric")
        if metric is None:
            return 0.0
        return float(metric.throughput())

