from LibSignal.common.registry import Registry
from LibSignal.agent.dqn import DQNAgent
from federated.agent.base import BaseFLAgent
from generator import LaneVehicleGenerator

@Registry.register_model('fl_dqn')
class FL_DQNAgent(DQNAgent, BaseFLAgent):
    """
    Inherits all DQN logic (replay buffer, get_ob, get_action, train, remember)
    from LibSignal's DQNAgent, and implements the FL interface.
    """
    def __init__(self, world, rank):
        super().__init__(world, rank)
        # Set generator to measure seconds of waiting time, not car count
        self.reward_generator = LaneVehicleGenerator(
            self.world, self.inter, ["lane_waiting_time_count"],
            in_only=True, average='all', negative=False
        )

    
    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)
        self.target_model.load_state_dict(weights)

    def get_reward(self):
        wait_time = self.reward_generator.generate()
        return - 0.01 * (wait_time ** 2)