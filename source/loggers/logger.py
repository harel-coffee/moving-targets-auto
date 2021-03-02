class Logger(object):
    """docstring for Learner"""

    def __init__(self):
        super(Logger, self).__init__()
        self.results = {}

    def on_process_start(self):
        pass

    def on_process_end(self):
        pass

    def on_pretraining_start(self):
        pass

    def on_pretraining_end(self):
        pass

    def on_iteration_start(self, idx):
        pass

    def on_iteration_end(self, idx):
        pass

    def on_training_start(self):
        pass

    def on_training_end(self):
        pass

    def on_adjustment_start(self):
        pass

    def on_adjustment_end(self, labels):
        pass
