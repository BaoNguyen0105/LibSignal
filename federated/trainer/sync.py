from common.registry import Registry
from federated.trainer.base import BaseFLTrainer
from federated.selection import build_selector
from federated.aggregation import build_aggregator
from federated.compression import build_compressor


@Registry.register_trainer('fl_tsc')
class SyncFLTrainer(BaseFLTrainer):
    """Synchronous Round-based Federated Learning Trainer."""
    def __init__(self, logger, gpu=0, cpu=False, name="fl_tsc"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        # 1. Pluggable strategies configured via YAML trainer settings
        trainer_param = Registry.mapping['trainer_mapping']['setting'].param
        self.selector = build_selector(trainer_param.get("selection", {}))
        self.aggregator = build_aggregator(trainer_param.get("aggregation", {}))
        self.compressor = build_compressor(trainer_param.get("compression", {}))

        # 2. Build clients using the configured compressor
        self.create_clients()
        self.logger.info(f"[FL] SyncFLTrainer ready (aggregation every {self.aggregate_interval} episodes).")

    def federated_round(self):
        client_weights = []
        sample_counts = []
        active_clients = []
        for client in self.selected_clients:
            compressed = client.get_weights()
            if compressed is None:
                self.logger.info(f"[FL] Client {client.get_client_id()} dropped update.")
                continue

            client_weights.append(self.compressor.decompress(compressed))
            sample_counts.append(client.get_sample_count())
            active_clients.append(client)

        if len(client_weights) == 0:
            self.logger.warning("[FL] All clients dropped updates this round.")
            return

        global_weights = self.aggregator.aggregate(
            client_weights, sample_counts, clients=active_clients
        )

        for client in self.clients.values():
            client.set_weights(global_weights)

    def train(self):
        self.logger.info(f"Start Training for {self.episodes} episodes...")
        for e in range(self.episodes):
            if e % self.aggregate_interval == 0:
                self.selected_clients = self.selector.select(list(self.clients.values()))

            self.train_episode(e)  # Inherited from BaseFLTrainer!

            if (e + 1) % self.aggregate_interval == 0:
                self.federated_round()

        [ag.save_model(e=self.episodes) for ag in self.agents]
