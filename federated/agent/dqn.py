from common.registry import Registry
from agent.dqn import DQNAgent
from federated.agent.base import BaseFLAgent
from generator import LaneVehicleGenerator
import random
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_


@Registry.register_model('fl_dqn')
class FL_DQNAgent(DQNAgent, BaseFLAgent):
    """
    Inherits all DQN logic (replay buffer, get_ob, get_action, train, remember)
    from LibSignal's DQNAgent, and implements the FL interface.
    """
    def __init__(self, world, rank):
        super().__init__(world, rank)
        self.round_sample_count = 0
        self._init_reward_generator()

    def _init_reward_generator(self):
        # Set generator to measure seconds of waiting time, not car count
        self.reward_generator = LaneVehicleGenerator(
            self.world, self.inter, ["lane_waiting_time_count"],
            in_only=True, average='all', negative=False
        )

    def reset(self):
        super().reset()
        self._init_reward_generator()

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        super().remember(last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key)
        # Track vehicle volume + transition count for sample-weighted FedAvg
        self.round_sample_count += max(1, int(np.sum(last_obs)))

    def train(self):
        samples = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions = self._batchwise(samples)
        out = self.target_model(b_tp, train=False)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        q_values = self.model(b_t, train=True).gather(1, actions.view(-1, 1)).squeeze(1)
        loss = self.criterion(q_values, target) + self.regularization_loss()
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.clone().detach().numpy()

    def get_weights(self):
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        cloned = {k: v.detach().clone() for k, v in weights.items()}
        self.model.load_state_dict(cloned)
        self.update_target_network()
        self.reset_sample_count()

    def get_reward(self):
        wait_time = np.squeeze(self.reward_generator.generate())
        return float(- 0.01 * (wait_time ** 2))
