import os
import numpy as np
import torch
from common.registry import Registry
from agent.frap import FRAP_DQNAgent
from federated.agent.base import BaseFLAgent


@Registry.register_model('fl_frap')
class FL_FRAPAgent(FRAP_DQNAgent, BaseFLAgent):
    """
    Inherits all FRAP logic (rotation-invariant phase competition network)
    from LibSignal's FRAP_DQNAgent, and implements the FL interface.
    """
    def __init__(self, world, rank):
        FRAP_DQNAgent.__init__(self, world, rank)
        BaseFLAgent.__init__(self)

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        super().remember(last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key)
        # Track vehicle volume + transition count for sample-weighted FedAvg
        self.round_sample_count += max(1, int(np.sum(last_obs)))

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

    def get_id(self):
        return str(self.id)

