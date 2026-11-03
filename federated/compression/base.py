from abc import ABC, abstractmethod


class BaseCompressor(ABC):
    """Abstract interface for communication compression."""

    def get_compression_ratio(self, **kwargs) -> float:
        """
        Returns the payload size ratio in (0.0, 1.0] relative to uncompressed FP32.
        Used by Client/Profile to scale transmission latency accordingly.
        """
        return 1.0

    @abstractmethod
    def compress(self, weights, **kwargs):
        """Compresses client weights before sending to server."""
        raise NotImplementedError

    @abstractmethod
    def decompress(self, compressed_weights, **kwargs):
        """Decompresses received weights back into PyTorch state_dict."""
        raise NotImplementedError
