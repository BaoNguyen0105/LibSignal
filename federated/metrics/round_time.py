from federated.metrics.base import BaseFLMetric


class RoundTimeMetric(BaseFLMetric):
    """
    Synchronous Bottleneck Round Completion Time ("round_time_ms").
    In synchronous FL, the server waits for the slowest active client:
        T_round = max_{i in active_clients} client_i.get_round_time()
    """
    name = "round_time_ms"
    scope = "round"

    def __init__(self, base_compute_ms: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self.base_compute_ms = float(base_compute_ms)

    def compute(self, **context) -> float:
        active_clients = context.get("active_clients") or []
        if not active_clients:
            return 0.0
        return float(max(c.get_round_time(self.base_compute_ms) for c in active_clients))

