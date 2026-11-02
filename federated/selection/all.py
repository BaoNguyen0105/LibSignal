from federated.selection.base import BaseClientSelector


class AllSelector(BaseClientSelector):
    """Selects 100% of clients (Full Participation)."""
    
    def __init__(self, **kwargs):
        pass

    def select(self, clients, **kwargs):
        # Simply return all clients!
        return clients