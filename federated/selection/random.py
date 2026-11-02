import random
from federated.selection.base import BaseClientSelector

class RandomSelector(BaseClientSelector):
    def __init__(self, fraction=0.5):
        self.fraction = fraction  # e.g., select 50% of intersections

    def select(self, clients, **kwargs):
        k = max(1, int(len(clients) * self.fraction))
        return random.sample(clients, k)