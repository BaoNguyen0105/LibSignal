import os
import numpy as np
import torch
from common.registry import Registry
from agent.sac import SACAgent
from federated.agent.base import BaseFLAgent


@Registry.register_model('fl_sac')
class FL_SACAgent(SACAgent, BaseFLAgent):
    """
    Inherits all Discrete Soft Actor-Critic logic (actor, V-critic, twin Q-critics)
    from LibSignal's SACAgent, and implements the FL interface.
    """
    def __init__(self, world, rank):
        SACAgent.__init__(self, world, rank)
        BaseFLAgent.__init__(self)

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        super().remember(last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key)
        # Track vehicle volume + transition count for sample-weighted FedAvg
        self.round_sample_count += max(1, int(np.sum(last_obs)))

    def get_weights(self):
        weights = {}
        for prefix, net in (
            ('actor', self.actor_net),
            ('v', self.v_evaluate_net),
            ('q0', self.q0_net),
            ('q1', self.q1_net),
        ):
            for k, v in net.state_dict().items():
                weights[f'{prefix}.{k}'] = v.detach().clone()
        return weights

    def set_weights(self, weights):
        for prefix, net in (
            ('actor', self.actor_net),
            ('v', self.v_evaluate_net),
            ('q0', self.q0_net),
            ('q1', self.q1_net),
        ):
            sub_dict = {
                k[len(prefix) + 1:]: v.detach().clone()
                for k, v in weights.items()
                if k.startswith(f'{prefix}.')
            }
            net.load_state_dict(sub_dict)
        self.update_target_network()
        self.reset_sample_count()

    def load_model(self, e):
        model_name = os.path.join(
            Registry.mapping['logger_mapping']['path'].path, 'model', f'{e}_{self.rank}.pt'
        )
        state_dict = torch.load(model_name, weights_only=True)
        self.actor_net.load_state_dict(state_dict)

    def get_id(self):
        return str(self.id)

