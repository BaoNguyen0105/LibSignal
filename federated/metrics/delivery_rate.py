from federated.metrics.base import BaseFLMetric


class DeliveryRateMetric(BaseFLMetric):
    """
    Packet Delivery Success Rate ("delivery_rate").
    Ratio of clients that successfully uploaded their weights (`active_clients`)
    to the number of clients scheduled for the round (`selected_clients`).
    """
    name = "delivery_rate"
    scope = "round"

    def compute(self, **context) -> float:
        selected_clients = context.get("selected_clients") or []
        active_clients = context.get("active_clients") or []
        if not selected_clients:
            return 0.0
        return float(len(active_clients)) / float(len(selected_clients))

