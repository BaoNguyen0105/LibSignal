from abc import ABC, abstractmethod


class BaseClientSelector(ABC):
    """Abstract interface for all client selection algorithms."""
    
    @abstractmethod
    def select(self, clients, **kwargs):
        """Takes a list of all available clients and returns the selected subset."""
        raise NotImplementedError