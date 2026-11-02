import os
import numpy as np
import torch
from common.registry import Registry
from federated.trainer.base import BaseFLTrainer
from federated.selection import AllSelector
from federated.aggregation import FedAvgAggregator
from federated.compression import NoCompression


@Registry.register_trainer('fl_tsc')
class SyncFLTrainer(BaseFLTrainer):
    """
    Synchronous Round-based Federated Learning Trainer.
    Runs local training steps on each intersection, and periodically averages weights.
    """
    def __init__(self, logger, gpu=0, cpu=False, name="fl_tsc"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        # 1. Modular FL components (customizable / pluggable)
        self.selector = AllSelector()
        self.aggregator = FedAvgAggregator(weighted=False)
        self.compressor = NoCompression()

        self.logger.info(f"[FL] Initialized SyncFLTrainer with aggregation every {self.aggregate_interval} episodes.")

    def federated_round(self):
        """Executes one round of Selection -> Collect -> Compress -> FedAvg -> Broadcast."""
        self.logger.info("========== [FL] Starting Federated Round ==========")

        # 1. SELECT: Pick which clients participate
        selected_clients = self.selector.select(self.agents)
        self.logger.info(f"[FL] Selected {len(selected_clients)} / {len(self.agents)} clients.")

        # 2. COLLECT & COMPRESS: Gather weights from clients that did not drop
        client_weights = []
        sample_counts = []
        for client in selected_clients:
            client_id = client.get_client_id()
            profile = self.profiles[client_id]

            # Simulate network packet loss
            if profile.is_dropped():
                self.logger.info(f"[FL] Client {client_id} dropped update due to network loss.")
                continue

            weights = client.get_weights()
            # Compress then decompress (simulating transmission over the network)
            decompressed = self.compressor.decompress(self.compressor.compress(weights))
            client_weights.append(decompressed)
            sample_counts.append(client.get_sample_count())

        # Defensive check: if all clients dropped updates
        if len(client_weights) == 0:
            self.logger.warning("[FL] No client updates received this round (all dropped). Skipping aggregation.")
            return

        # 3. AGGREGATE: Run FedAvg
        global_weights = self.aggregator.aggregate(client_weights, sample_counts=sample_counts)

        # 4. BROADCAST: Update all clients with the new global model
        for client in self.agents:
            client.set_weights(global_weights)

        self.logger.info(f"========== [FL] Global Model Broadcast to {len(self.agents)} Clients Complete ==========")

    def train(self):
        """
        Executes the CityFlow simulation across episodes, triggering federated_round()
        every `self.aggregate_interval` episodes.
        """
        total_decision_num = 0
        flush = 0
        self.logger.info(f"Start Federated Training for {self.episodes} episodes...")

        for e in range(self.episodes):
            self.metric.clear()
            last_obs = self.env.reset()

            for a in self.agents:
                a.reset()

            # Optional replay recording for CityFlow visualizer
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)

            episode_loss = []
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents])

                    if total_decision_num > self.learning_start:
                        actions = []
                        for idx, ag in enumerate(self.agents):
                            actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))
                        actions = np.stack(actions)
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents])

                    actions_prob = []
                    for idx, ag in enumerate(self.agents):
                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)
                    self.metric.update(rewards)

                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])
                    for idx, ag in enumerate(self.agents):
                        ag.remember(
                            last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx],
                            rewards[idx], obs[idx], cur_phase[idx], dones[idx],
                            f'{e}_{i//self.action_interval}_{ag.id}'
                        )
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0

                    total_decision_num += 1
                    last_obs = obs

                # Local training step for each agent
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                    cur_loss_q = np.stack([ag.train() for ag in self.agents])
                    episode_loss.append(cur_loss_q)

                # Local target network update
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents]

                if all(dones):
                    break

            mean_loss = np.mean(np.array(episode_loss)) if len(episode_loss) > 0 else 0

            # Log metrics (rewards, queues, travel times)
            self.writeLog("TRAIN", e, self.metric.real_average_travel_time(),
                          mean_loss, self.metric.rewards(), self.metric.queue(),
                          self.metric.delay(), self.metric.throughput())

            self.logger.info("step:{}/{}, q_loss:{:.4f}, rewards:{:.2f}, queue:{:.2f}, delay:{:.2f}, throughput:{}".format(
                i, self.steps, mean_loss, self.metric.rewards(), self.metric.queue(),
                self.metric.delay(), int(self.metric.throughput())))

            if e % self.save_rate == 0:
                [ag.save_model(e=e) for ag in self.agents]

            self.logger.info("episode:{}/{}, real avg travel time:{:.2f}".format(
                e, self.episodes, self.metric.real_average_travel_time()))

            if self.test_when_train:
                self.train_test(e)

            # =========================================================
            # FEDERATED AGGREGATION: Trigger round every interval
            # =========================================================
            if (e + 1) % self.aggregate_interval == 0:
                self.federated_round()

        # Save final models after all episodes complete
        [ag.save_model(e=self.episodes) for ag in self.agents]
