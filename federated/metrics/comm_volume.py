from federated.metrics.base import BaseFLMetric


class CommVolumeMetric(BaseFLMetric):
    """
    Round Communication Volume ("comm_volume_mb").
    Calculates total compressed model payload (in MB) uploaded by active clients
    during the federated round.
    """
    name = "comm_volume_mb"
    scope = "round"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_comm_mb = 0.0

    def compute(self, **context) -> float:
        active_clients = context.get("active_clients") or []
        round_mb = float(sum(c.get_payload_size_mb() for c in active_clients))
        self.total_comm_mb += round_mb
        return round_mb

    def reset(self) -> None:
        self.total_comm_mb = 0.0

