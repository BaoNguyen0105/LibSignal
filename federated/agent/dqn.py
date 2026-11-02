import os
import random
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from common.registry import Registry
from agent.dqn import DQNAgent
from federated.agent.base import BaseFLAgent


@Registry.register_model('fl_dqn')
class FL_DQNAgent(DQNAgent, BaseFLAgent):
    """
    Inherits all DQN logic (replay buffer, get_ob, get_action, train, remember, get_reward)
    from LibSignal's DQNAgent, and implements the FL interface.
    """
    def __init__(self, world, rank):
        super().__init__(world, rank)
        self.round_sample_count = 0

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
        loss = (self.criterion(q_values, target) / self.action_space.n) + self.regularization_loss()
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

    def load_model(self, e):
        model_name = os.path.join(
            Registry.mapping['logger_mapping']['path'].path, 'model', f'{e}_{self.rank}.pt'
        )
        state_dict = torch.load(model_name, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.target_model.load_state_dict(state_dict)
