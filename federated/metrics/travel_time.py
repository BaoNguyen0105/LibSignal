from federated.metrics.base import BaseFLMetric


class TravelTimeMetric(BaseFLMetric):
    """
    LibSignal Original Metric: Average Travel Time ("travel_time").
    Average time (seconds) vehicles spend traveling in the network.
    """
    name = "travel_time"
    scope = "episode"

    def compute(self, **context) -> float:
        metric = context.get("metric")
        if metric is None:
            return 0.0
        return float(metric.real_average_travel_time())

