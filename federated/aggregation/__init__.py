from .base import BaseAggregator
from .fedavg import FedAvgAggregator

AGGREGATOR_REGISTRY = {
    "fedavg": FedAvgAggregator,
}


def build_aggregator(aggregator_cfg: dict | None = None) -> BaseAggregator:
    """Builds the configured federated aggregator strategy from YAML settings."""
    aggregator_cfg = aggregator_cfg or {}
    agg_type = aggregator_cfg.get("type", "fedavg")
    agg_cls = AGGREGATOR_REGISTRY[agg_type]
    params = {k: v for k, v in aggregator_cfg.items() if k != "type"}
    return agg_cls(**params)
