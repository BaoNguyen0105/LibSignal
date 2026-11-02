from abc import ABC, abstractmethod
from trainer.tsc_trainer import TSCTrainer
from common.registry import Registry
from federated.profile import build_profile


class BaseFLTrainer(TSCTrainer, ABC):
    """
    Abstract Base Class for all Federated Learning Trainers.
    Inherits all CityFlow simulation and evaluation machinery from TSCTrainer.
    """
    def __init__(self, logger, gpu=0, cpu=False, name="fl_tsc"):
        # Instantiates world, agents, metric, and env via TSCTrainer
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        # 1. Load trainer params from LibSignal registry
        trainer_param = Registry.mapping['trainer_mapping']['setting'].param
        self.aggregate_interval = trainer_param.get('aggregate_interval', 20)
        profile_cfg = trainer_param.get("client_profiles", {})

        # 2. Physical network profiles managed directly by the Trainer
        self.profiles = {
            ag.get_client_id(): build_profile(ag.get_client_id(), profile_cfg)
            for ag in self.agents
        }

        # 3. Pluggable FL components
        self.selector = None
        self.aggregator = None
        self.compressor = None

    @abstractmethod
    def federated_round(self):
        """Must implement how one round of federated communication is executed."""
        raise NotImplementedError