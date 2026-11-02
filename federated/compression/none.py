from federated.compression.base import BaseCompressor


class NoCompression(BaseCompressor):
    """Standard uncompressed full-precision (FP32) weights."""

    def __init__(self, **kwargs):
        pass

    def compress(self, weights, **kwargs):
        # Return weights completely untouched!
        return weights

    def decompress(self, compressed_weights, **kwargs):
        # Return weights completely untouched!
        return compressed_weights
