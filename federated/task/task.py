from common.registry import Registry
from task.task import BaseTask

@Registry.register_task('fl_tsc')
class FLTSCTask(BaseTask):
    """Self-contained task runner for federated traffic signal control."""
    def run(self):
        self.trainer.train()