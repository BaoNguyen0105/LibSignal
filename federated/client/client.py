import time


class Client:
    """Fuses Hardware (Profile) and Software (Agent) into one physical edge box."""
    def __init__(self, agent, profile, compressor):
        self.agent = agent
        self.profile = profile
        self.compressor = compressor
        self.agent.set_buffer_size(self.profile.buffer_size)

    def get_weights(self, compress: bool = True, real_sleep: bool = False):
        """Returns weights (compressed or raw), or None if dropped."""
        if self.profile.is_dropped():
            return None
        weights = self.agent.get_weights()
        if real_sleep:
            time.sleep(self.profile.get_latency() / 1000.0)
        return self.compressor.compress(weights) if compress else weights

    def set_weights(self, weights):
        """Updates the local agent with the global model."""
        self.agent.set_weights(weights)

    def get_sample_count(self):
        return self.agent.get_sample_count()

    def reset_sample_count(self):
        self.agent.reset_sample_count()

    def get_client_id(self):
        return self.agent.get_id()

    def get_latency(self):
        return self.profile.get_latency()
