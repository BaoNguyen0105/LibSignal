import os
from abc import ABC, abstractmethod
import numpy as np
from trainer.tsc_trainer import TSCTrainer
from common.registry import Registry
from federated.profile import build_profile
from federated.client import Client
from federated.metrics import build_metrics, compute_scoped_metrics


class BaseFLTrainer(TSCTrainer, ABC):
    """
    Abstract Base Class for all Federated Learning Trainers.
    Inherits all CityFlow simulation and evaluation machinery from TSCTrainer.
    """
    def __init__(self, logger, gpu=0, cpu=False, name="fl_tsc"):
        self.profiles = {}
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        # 1. Load trainer params from LibSignal registry
        trainer_param = Registry.mapping['trainer_mapping']['setting'].param
        self.aggregate_interval = trainer_param.get('aggregate_interval', 20)

        # 2. Strategy placeholders (subclasses initialize these)
        self.selector = None
        self.aggregator = None
        self.compressor = None
        self.clients = {}
        self.selected_clients = []
        self.global_weights = None
        self.fl_metrics = build_metrics(trainer_param.get("fl_metrics"))

        # 3. Simulation tracking state
        self.total_decision_num = 0
        self.flush = 0

    def create_clients(self):
        """Builds hardware profiles, broadcasts a common initial global model, and constructs Clients."""
        trainer_param = Registry.mapping['trainer_mapping']['setting'].param
        profile_cfg = trainer_param.get("client_profiles", {})
        self.profiles = {
            str(inter_id): build_profile(str(inter_id), profile_cfg)
            for inter_id in self.world.intersection_ids
        }

        initial_weights = self.agents[0].get_weights()
        for ag in self.agents:
            ag.set_weights(initial_weights)
        self.global_weights = initial_weights

        self.clients = {
            ag.get_id(): Client(ag, self.profiles[ag.get_id()], self.compressor)
            for ag in self.agents
        }
        self.selected_clients = list(self.clients.values())

    def on_step_end(self):
        """Hook called at the end of every simulation step (e.g. for Async or Step-based FL)."""
        pass

    def train_episode(self, e: int):
        """Runs one full CityFlow episode (all steps, metrics, and local agent updates)."""
        self.metric.clear()
        last_obs = self.env.reset()

        for a in self.agents:
            a.reset()

        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay and e % self.save_rate == 0:
                replay_rel = os.path.join(self.replay_file_dir, f"episode_{e}.txt")
                replay_abs = os.path.join(self.dir, replay_rel)
                if os.path.exists(replay_abs):
                    os.remove(replay_abs)
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(replay_rel)
            else:
                self.env.eng.set_save_replay(False)

        episode_loss = []
        i = 0
        while i < self.steps:
            if i % self.action_interval == 0:
                last_phase = np.stack([ag.get_phase() for ag in self.agents])

                if self.total_decision_num > self.learning_start:
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
                self.flush += 1
                if self.flush == self.buffer_size - 1:
                    self.flush = 0

                self.total_decision_num += 1
                last_obs = obs

            # Local training step for selected clients
            if self.total_decision_num > self.learning_start and \
                    self.total_decision_num % self.update_model_rate == self.update_model_rate - 1:
                step_losses = [c.agent.train() for c in self.selected_clients]
                if step_losses:
                    episode_loss.append(np.mean(np.stack(step_losses)))

            # Target network update for selected clients
            if self.total_decision_num > self.learning_start and \
                    self.total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                [c.agent.update_target_network() for c in self.selected_clients]

            self.on_step_end()

            if all(dones):
                break

        mean_loss = float(np.mean(np.array(episode_loss))) if len(episode_loss) > 0 else 0.0

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

        ep_metrics = compute_scoped_metrics(self.fl_metrics, "episode", metric=self.metric)
        if ep_metrics:
            ep_str = ", ".join(f"{k}:{v:.4f}" for k, v in ep_metrics.items())
            self.logger.info(f"[FL Episode Metrics] {ep_str}")

        for j in range(len(self.world.intersections)):
            self.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(
                j, self.metric.lane_rewards()[j], self.metric.lane_queue()[j]))

        if self.test_when_train:
            self.train_test(e)

    def test(self, drop_load=True):
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow' and self.save_replay:
            final_abs = os.path.join(self.dir, self.replay_file_dir, "final.txt")
            if os.path.exists(final_abs):
                os.remove(final_abs)
        res_metric = super().test(drop_load=drop_load)
        ep_metrics = compute_scoped_metrics(self.fl_metrics, "episode", metric=self.metric)
        if ep_metrics:
            ep_str = ", ".join(f"{k}:{v:.4f}" for k, v in ep_metrics.items())
            self.logger.info(f"[FL Final Test Metrics] {ep_str}")
        return res_metric

    @abstractmethod
    def federated_round(self):
        """Must implement how one round of federated communication is executed."""
        raise NotImplementedError
