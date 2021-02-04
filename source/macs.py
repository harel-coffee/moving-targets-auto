"""
Module to define abstract classes to be used.
"""


class Learner(object):
    """docstring for Learner"""

    def __init__(self):
        super(Learner, self).__init__()

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass


class Master(object):
    """docstring for Learner"""

    def __init__(self, **kwargs):
        super(Master, self).__init__()

    def adjust_targets(self, y, p, alpha, beta, use_prob):
        pass

    def score_info(self, **kwargs):
        pass

    def cst_info(self, **kwargs):
        pass


class Logger(object):
    """docstring for Learner"""

    def __init__(self):
        super(Logger, self).__init__()

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


class MACS(object):
    """docstring for ModelAgnosticConstrainedClassifier"""

    def __init__(self, learner, master, logger=Logger()):
        super(MACS, self).__init__()
        self.learner = learner
        self.master = master
        self.logger = logger

    def fit(self, x, y,
            niterations,
            alpha,
            beta,
            initial_step,
            use_prob=False):
        # Print stuff
        self.logger.on_process_start()

        # Handle pretraining
        if initial_step == 'pretraining':
            self.logger.on_pretraining_start()
            # Train with the original labels
            self.learner.fit(x, y)
            self.logger.on_pretraining_end()

        elif initial_step == 'projection':
            self.logger.on_pretraining_start()
            # Project targets on the feasible region.
            adj_y = self.master.adjust_targets(y, y.reshape(-1), 1e6, beta, use_prob)
            self.learner.fit(x, adj_y)
            self.logger.on_pretraining_end()
        else:
            raise ValueError("The method doesn't accept the initial step " + initial_step)

        # Start the main algorithm
        for it in range(niterations):
            self.logger.on_iteration_start(it)
            # Obtain the current predictions
            if alpha > 0:
                if not use_prob:
                    pred = self.learner.predict(x)
                else:
                    pred = self.learner.predict_proba(x)
            else:
                pred = None
            # Find an assignment of the targets that satisfies the constraints
            self.logger.on_adjustment_start()
            adj_y = self.master.adjust_targets(y, pred, alpha, beta, use_prob)
            if adj_y is None:
                print('ERROR: could not compute adjusted labels')
                return
            self.logger.on_adjustment_end(adj_y, it)
            # Train the learner with the new labels
            self.logger.on_training_start()
            self.learner.fit(x, adj_y)
            self.logger.on_training_end(it)
            self.logger.on_iteration_end(it)

        # Print stuff
        self.logger.on_process_end()

    def predict(self, x):
        self.learner.predict(x)

    def predict_proba(self, x):
        self.learner.predict_proba(x)
