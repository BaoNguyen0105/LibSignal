from common.metrics import Metrics
from .base import BaseFLMetric
from .travel_time import TravelTimeMetric
from .rewards import RewardsMetric
from .queue import QueueMetric
from .delay import DelayMetric
from .throughput import ThroughputMetric
from .max_queue import MaxQueueMetric
from .queue_fairness import QueueFairnessMetric
from .round_time import RoundTimeMetric
from .comm_volume import CommVolumeMetric
from .delivery_rate import DeliveryRateMetric
from .weight_drift import WeightDriftMetric

METRIC_REGISTRY = {
    # LibSignal original traffic metrics
    "travel_time": TravelTimeMetric,
    "rewards": RewardsMetric,
    "queue": QueueMetric,
    "delay": DelayMetric,
    "throughput": ThroughputMetric,
    # Spatial fairness traffic metrics
    "max_queue": MaxQueueMetric,
    "queue_fairness": QueueFairnessMetric,
    # Federated system & communication metrics
    "round_time": RoundTimeMetric,
    "comm_volume": CommVolumeMetric,
    "delivery_rate": DeliveryRateMetric,
    "weight_drift": WeightDriftMetric,
}

DEFAULT_METRICS = list(METRIC_REGISTRY.keys())


def build_metrics(metrics_cfg: list | dict | None = None) -> list[BaseFLMetric]:
    """Builds the list of active FL and traffic metrics from YAML settings."""
    if metrics_cfg is None:
        return [METRIC_REGISTRY[name]() for name in DEFAULT_METRICS]

    if isinstance(metrics_cfg, list):
        instances = []
        for item in metrics_cfg:
            if isinstance(item, str) and item in METRIC_REGISTRY:
                instances.append(METRIC_REGISTRY[item]())
            elif isinstance(item, dict):
                m_type = item.get("type")
                if m_type in METRIC_REGISTRY:
                    params = {k: v for k, v in item.items() if k != "type"}
                    instances.append(METRIC_REGISTRY[m_type](**params))
        return instances

    if isinstance(metrics_cfg, dict):
        instances = []
        for name, cfg in metrics_cfg.items():
            if name not in METRIC_REGISTRY:
                continue
            if isinstance(cfg, dict):
                if cfg.get("enabled", True):
                    params = {k: v for k, v in cfg.items() if k != "enabled"}
                    instances.append(METRIC_REGISTRY[name](**params))
            elif bool(cfg):
                instances.append(METRIC_REGISTRY[name]())
        return instances

    return [METRIC_REGISTRY[name]() for name in DEFAULT_METRICS]


def compute_scoped_metrics(
    metrics: list[BaseFLMetric],
    scope: str,
    **context,
) -> dict[str, float]:
    """Evaluates all metrics matching `scope` ('episode' or 'round')."""
    return {
        m.name: m.compute(**context)
        for m in metrics
        if m.scope == scope
    }
