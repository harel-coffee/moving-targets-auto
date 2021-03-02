import wandb
import time

from source import macs

PROJECT = 'test'
ENTITY = 'giuluck'

class WandBLogger(macs.Logger):
    """docstring for WandBLogger"""

    def __init__(self, learner, master, x, y, x_test=None, y_test=None, params=None, run='test'):
        super(WandBLogger, self).__init__()
        # wandb.login(key=None)
        wandb.init(project=PROJECT, entity=ENTITY, name=run)
        self.learner = learner
        self.master = master
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.logs = {} if params is None else params
        self.results = {}

    def _append_info(self, x, y, p, name):
        for info in [self.master.cst_info(x, p), self.master.score_info(y, p)]:
            for k, v in info.items():
                self.logs[f'{name}{k}'] = v

    def on_process_end(self):
        wandb.finish()

    def on_pretraining_start(self):
        self.on_iteration_start(-1)

    def on_pretraining_end(self):
        self.on_iteration_end(-1)

    def on_iteration_start(self, idx):
        self.logs['iteration'] = idx + 1
        self.logs['time/iteration'] = time.time()

    def on_iteration_end(self, idx):
        self.logs['time/iteration'] = time.time() - self.logs['time/iteration']
        wandb.log(self.logs)
        self.logs = {}

    def on_training_start(self):
        self.logs['time/learner'] = time.time()

    def on_training_end(self):
        self.logs['time/learner'] = time.time() - self.logs['time/learner']
        self._append_info(self.x, self.y, self.learner.predict(self.x), 'learner-train/')
        if self.x_test is not None:
            self._append_info(self.x_test, self.y_test, self.learner.predict(self.x_test), 'learner-test/')

    def on_adjustment_start(self):
        self.logs['time/master'] = time.time()

    def on_adjustment_end(self, labels, it=None):
        self.logs['time/master'] = time.time() - self.logs['time/master']
        self._append_info(self.x, self.y, labels, 'master/')
