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
from common.registry import Registry
from common import interface
from common.utils import *
from utils.logger import setup_logging, load_config, get_output_file_path


# Arguments parser
parser = argparse.ArgumentParser(description='Run Federated Experiment')
parser.add_argument('--thread_num', type=int, default=4)
parser.add_argument('--ngpu', type=str, default="-1")
parser.add_argument('--prefix', type=str, default='test')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--debug', type=bool, default=True)
parser.add_argument('--interface', type=str, default="libsumo", choices=['libsumo', 'traci'])
parser.add_argument('--delay_type', type=str, default="apx", choices=['apx', 'real'])

parser.add_argument('-t', '--task', type=str, default="fl_tsc", help="task type to run")
parser.add_argument('-a', '--agent', type=str, default="fl_dqn", help="agent type")
parser.add_argument('-w', '--world', type=str, default="cityflow", choices=['cityflow', 'sumo'])
parser.add_argument('-n', '--network', type=str, default="cityflow4x4", help="network name")
parser.add_argument('-d', '--dataset', type=str, default='onfly')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

logging_level = logging.DEBUG if args.debug else logging.INFO


def build_fl_config(pArgs):
    """Checks federated/configs/ first; falls back to root configs/ if needed."""
    fl_config_path = os.path.join('federated', 'configs', pArgs.task, f'{pArgs.agent}.yml')
    if os.path.exists(fl_config_path):
        config, duplicates = load_config(fl_config_path)
    else:
        config, duplicates = load_config(os.path.join('configs', pArgs.task, f'{pArgs.agent}.yml'))
    config.update({'command': pArgs.__dict__})
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
        self.trainer = Registry.mapping['trainer_mapping'][task_name](logger)
        self.task = Registry.mapping['task_mapping'][task_name](self.trainer)
        start_time = time.time()
        self.task.run()
        logger.info(f"Total time taken: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    runner = FLRunner(args)
    runner.run()