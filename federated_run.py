import os
import sys
import time
import argparse
import logging

import task
import trainer
import agent
import dataset
import federated  # Registers all FL models, trainers, tasks, and profiles
from federated.trainer import TRAINER_REGISTRY
from federated.selection import SELECTOR_REGISTRY
from federated.aggregation import AGGREGATOR_REGISTRY
from federated.compression import COMPRESSOR_REGISTRY
from federated.profile import PROFILE_REGISTRY
from common.registry import Registry
from common import interface
from common.utils import *
from utils.logger import setup_logging, load_config, get_output_file_path


# Arguments parser
parser = argparse.ArgumentParser(description='Run Federated Experiment')
parser.add_argument('--thread_num', type=int, default=8)
parser.add_argument('--ngpu', type=str, default="-1")
parser.add_argument('--prefix', type=str, default='test')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--debug', type=bool, default=True)
parser.add_argument('--interface', type=str, default="libsumo", choices=['libsumo', 'traci'])
parser.add_argument('--delay_type', type=str, default="apx", choices=['apx', 'real'])

# Core LibSignal task / agent / environment flags
parser.add_argument('-t', '--task', type=str, default="fl_tsc", help="task type to run")
parser.add_argument('-a', '--agent', type=str, default="fl_dqn", help="agent type (fl_dqn, fl_presslight, fl_frap, fl_sac)")
parser.add_argument('-w', '--world', type=str, default="cityflow", choices=['cityflow', 'sumo'])
parser.add_argument('-n', '--network', type=str, default="cityflow4x4", help="network name")
parser.add_argument('-d', '--dataset', type=str, default='onfly')
parser.add_argument('--test', action='store_true', help="evaluate saved model without training")

# Federated Learning & training schedule CLI overrides
parser.add_argument('-e', '--episodes', type=int, default=None, help="override total training episodes")
parser.add_argument('--trainer', type=str, default=None, choices=list(TRAINER_REGISTRY.keys()),
                    help="FL trainer paradigm (sync, async, semi_async, step)")
parser.add_argument('--aggregate_interval', type=int, default=None,
                    help="episodes per FL aggregation round")
parser.add_argument('--aggregate_steps', type=int, default=None,
                    help="decision steps per FL aggregation (for --trainer step)")

# Pluggable FL strategy CLI overrides
parser.add_argument('-s', '--selection', type=str, default=None, choices=list(SELECTOR_REGISTRY.keys()),
                    help="client selection strategy (all, random, fedcs, fair_fedcs, resource)")
parser.add_argument('--fraction', type=float, default=None,
                    help="fraction of clients selected per round (0.0 - 1.0)")
parser.add_argument('--fairness_weight', type=float, default=None,
                    help="UCB fairness weight for fedcs / fair_fedcs selector")

parser.add_argument('-g', '--aggregation', type=str, default=None, choices=list(AGGREGATOR_REGISTRY.keys()),
                    help="server aggregation strategy (fedavg, fednova, unbiased, fedavgm)")
parser.add_argument('--weighted', action='store_true', default=None,
                    help="enable traffic/sample-weighted aggregation")

parser.add_argument('-c', '--compression', type=str, default=None, choices=list(COMPRESSOR_REGISTRY.keys()),
                    help="weight compression strategy (none, quantization, topk, low_rank, adaptive)")
parser.add_argument('--bits', type=int, default=None,
                    help="quantization bit-width (for quantization compressor)")

parser.add_argument('-p', '--profile', type=str, default=None, choices=list(PROFILE_REGISTRY.keys()),
                    help="edge hardware & network profile (simple, wireless, markov, tiered)")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

logging_level = logging.DEBUG if args.debug else logging.INFO


def build_fl_config(pArgs):
    """Checks federated/configs/ first; falls back to fl_tsc or root configs/ if needed."""
    fl_config_path = os.path.join('federated', 'configs', pArgs.task, f'{pArgs.agent}.yml')
    fallback_fl_path = os.path.join('federated', 'configs', 'fl_tsc', f'{pArgs.agent}.yml')
    if os.path.exists(fl_config_path):
        config, duplicates = load_config(fl_config_path)
    elif os.path.exists(fallback_fl_path):
        config, duplicates = load_config(fallback_fl_path)
    else:
        config, duplicates = load_config(os.path.join('configs', pArgs.task, f'{pArgs.agent}.yml'))
    config.update({'command': pArgs.__dict__})

    trainer_cfg = config.setdefault('trainer', {})
    if pArgs.episodes is not None:
        trainer_cfg['episodes'] = pArgs.episodes
    if pArgs.trainer is not None:
        trainer_cfg['type'] = pArgs.trainer
    if pArgs.aggregate_interval is not None:
        trainer_cfg['aggregate_interval'] = pArgs.aggregate_interval
    if pArgs.aggregate_steps is not None:
        trainer_cfg['aggregate_steps'] = pArgs.aggregate_steps

    if pArgs.selection is not None:
        trainer_cfg.setdefault('selection', {})['type'] = pArgs.selection
    if pArgs.fraction is not None:
        trainer_cfg.setdefault('selection', {})['fraction'] = pArgs.fraction
    if pArgs.fairness_weight is not None:
        trainer_cfg.setdefault('selection', {})['fairness_weight'] = pArgs.fairness_weight

    if pArgs.aggregation is not None:
        trainer_cfg.setdefault('aggregation', {})['type'] = pArgs.aggregation
    if pArgs.weighted is not None:
        trainer_cfg.setdefault('aggregation', {})['weighted'] = True

    if pArgs.compression is not None:
        trainer_cfg.setdefault('compression', {})['type'] = pArgs.compression
    if pArgs.bits is not None:
        trainer_cfg.setdefault('compression', {})['bits'] = pArgs.bits

    if pArgs.profile is not None:
        trainer_cfg.setdefault('client_profiles', {})['type'] = pArgs.profile

    if pArgs.test:
        config['model']['train_model'] = False
        config['model']['test_model'] = True
        config['model']['load_model'] = True
        config['world']['saveReplay'] = True
    return config, duplicates


class FLRunner:
    def __init__(self, pArgs):
        self.config, self.duplicate_config = build_fl_config(pArgs)
        self.config_registry()

    def config_registry(self):
        interface.Command_Setting_Interface(self.config)
        interface.Logger_param_Interface(self.config)
        interface.World_param_Interface(self.config)
        if self.config['model'].get('graphic', False):
            param = Registry.mapping['world_mapping']['setting'].param
            roadnet_path = param['dir'] + param['roadnetFile'] if self.config['command']['world'] in ['cityflow', 'sumo'] else param['road_file_addr']
            interface.Graph_World_Interface(roadnet_path)
        interface.Logger_path_Interface(self.config)

        if not os.path.exists(Registry.mapping['logger_mapping']['path'].path):
            os.makedirs(Registry.mapping['logger_mapping']['path'].path)
        interface.Trainer_param_Interface(self.config)
        interface.ModelAgent_param_Interface(self.config)

    def run(self):
        logger = setup_logging(logging_level)
        task_name = Registry.mapping['command_mapping']['setting'].param['task']
        trainer_param = Registry.mapping['trainer_mapping']['setting'].param
        trainer_type = trainer_param.get('type', task_name)
        if trainer_type in TRAINER_REGISTRY:
            self.trainer = federated.trainer.build_fl_trainer(trainer_type, logger)
        else:
            self.trainer = Registry.mapping['trainer_mapping'][task_name](logger)
        self.task = Registry.mapping['task_mapping'][task_name](self.trainer)
        start_time = time.time()
        self.task.run()
        logger.info(f"Total time taken: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    runner = FLRunner(args)
    runner.run()
