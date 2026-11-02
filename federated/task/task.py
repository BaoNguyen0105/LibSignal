from common.registry import Registry
from task.task import BaseTask


@Registry.register_task('fl_tsc')
class FLTSCTask(BaseTask):
    """Self-contained task runner for federated traffic signal control."""
    def run(self):
        model_param = Registry.mapping['model_mapping']['setting'].param
        if model_param['train_model']:
            self.trainer.train()
        if model_param['test_model']:
            self.trainer.test(drop_load=not model_param['load_model'])
