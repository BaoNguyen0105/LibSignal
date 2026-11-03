import time


class Client:
    """Fuses Hardware (Profile) and Software (Agent) into one physical edge box."""
    def __init__(self, agent, profile, compressor):
        self.agent = agent
        self.profile = profile
        self.compressor = compressor
        self.base_weights = None
        self.agent.set_buffer_size(self.profile.buffer_size)

    def get_weights(self, compress: bool = True, real_sleep: bool = False):
        """Returns weights (compressed or raw), or None if dropped."""
        if self.profile.is_dropped():
            return None
        weights = self.agent.get_weights()
        if real_sleep:
            time.sleep(self.get_latency(uncompressed=not compress) / 1000.0)
        return self.compressor.compress(weights, client=self) if compress else weights

    def set_weights(self, weights):
        """Updates the local agent with the global model."""
        self.base_weights = {k: v.detach().clone() for k, v in weights.items()}
        self.agent.set_weights(weights)

    def get_sample_count(self):
        return self.agent.get_sample_count()

    def reset_sample_count(self):
        self.agent.reset_sample_count()

    def get_client_id(self):
        return self.agent.get_id()

    def step_profile(self) -> None:
        self.profile.step_state()

    def get_compression_ratio(self) -> float:
        if self.compressor is None:
            return 1.0
        return self.compressor.get_compression_ratio(client=self)

    def get_payload_size_mb(self, uncompressed: bool = False) -> float:
        ratio = 1.0 if uncompressed else self.get_compression_ratio()
        return self.profile.get_payload_size_mb(compression_ratio=ratio)

    def get_latency(self, uncompressed: bool = False) -> float:
        ratio = 1.0 if uncompressed else self.get_compression_ratio()
        return self.profile.get_latency(compression_ratio=ratio)

    def get_drop_rate(self) -> float:
        return self.profile.get_drop_rate()

    def get_compute_speed(self) -> float:
        return self.profile.get_compute_speed()

    def get_round_time(self, base_compute_ms: float = 100.0) -> float:
        ratio = self.get_compression_ratio()
        return self.profile.get_round_time(
            base_compute_ms=base_compute_ms,
            compression_ratio=ratio,
        )
