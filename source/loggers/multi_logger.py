from source.loggers.logger import Logger

class MultiLogger(Logger):
    """docstring for MultiLogger"""

    def __init__(self, loggers: list):
        super(MultiLogger, self).__init__()
        self.loggers = loggers

    def _update_loggers(self, routine):
        for log in self.loggers:
            routine(log)
            self.results.update(log.results)

    def on_process_start(self):
        self._update_loggers(lambda log: log.on_process_start())

    def on_process_end(self):
        self._update_loggers(lambda log: log.on_process_end())

    def on_pretraining_start(self):
        self._update_loggers(lambda log: log.on_pretraining_start())

    def on_pretraining_end(self):
        self._update_loggers(lambda log: log.on_pretraining_end())

    def on_iteration_start(self, idx):
        self._update_loggers(lambda log: log.on_iteration_start(idx))

    def on_iteration_end(self, idx):
        self._update_loggers(lambda log: log.on_iteration_end(idx))

    def on_training_start(self):
        self._update_loggers(lambda log: log.on_training_start())

    def on_training_end(self):
        self._update_loggers(lambda log: log.on_training_end())

    def on_adjustment_start(self):
        self._update_loggers(lambda log: log.on_adjustment_start())

    def on_adjustment_end(self, labels):
        self._update_loggers(lambda log: log.on_adjustment_end(labels))
