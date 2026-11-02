from abc import ABC, abstractmethod


class BaseCompressor(ABC):
    """Abstract interface for communication compression."""

    @abstractmethod
    def compress(self, weights):
        """Compresses client weights before sending to server."""
        raise NotImplementedError

    @abstractmethod
    def decompress(self, compressed_weights):
        """Decompresses received weights back into PyTorch state_dict."""
        raise NotImplementedError