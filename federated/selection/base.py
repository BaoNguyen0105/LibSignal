from abc import ABC, abstractmethod
from federated.client import Client

class BaseClientSelector(ABC):
    """Abstract interface for all client selection algorithms."""
    
    @abstractmethod
    def select(self, clients:list[Client], **kwargs):
        """Takes a list of all available clients and returns the selected subset."""
        raise NotImplementedError