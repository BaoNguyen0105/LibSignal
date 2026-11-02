import random
from federated.selection.base import BaseClientSelector


class RandomSelector(BaseClientSelector):
    def __init__(self, fraction: float = 0.5, **kwargs):
        self.fraction = max(0.0, min(1.0, float(fraction)))  # e.g., select 50% of intersections

    def select(self, clients, **kwargs):
        if not clients:
            return []
        k = max(1, int(len(clients) * self.fraction))
        return random.sample(clients, min(k, len(clients)))
