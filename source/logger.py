"""
Module to define the logger used to self.logger.info the results of each run.
"""

import pandas as pd
import sklearn
import time
import logging

from source import macs

"""
Logging verbosity:
    - 0: overall results
    - 1: specific results
    - 2: debug mode
"""
_verbosity = 0


class CustomLogger(macs.Logger):
    """docstring for CustomLogger"""

    def __init__(self, learner, master, x, y, x_test=None, y_test=None, nfold=0,
                 log_on_file=True, name='log'):
        super(CustomLogger, self).__init__()
        self.learner = learner
        self.master = master
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.name = name
        self.nfold = nfold

        self.results = dict()

        self.logger = logging.Logger(__name__)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)

        if log_on_file:
            file_handler = logging.FileHandler(f'{self.name}.log', 'a', 'utf-8')
            file_formatter = logging.Formatter(fmt='%(message)s',
                                               datefmt='%d/%m/%Y %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)

    def on_process_start(self):
        self.logger.info(f'### Processing fold {self.nfold}')
        self.logger.info('>>> Starting the MACS process')
        # Print constraint satisfaction info about the datasets
        _cost = self.master.cst_info(self.x, self.y)
        for k in _cost.keys():
            self.logger.info(f"{k} in the training set: {_cost[k]:.3f}")
        self.results['Initial train set'] = _cost

        if self.y_test is not None:
            _cost = self.master.cst_info(self.x_test, self.y_test)
            for k in _cost.keys():
                self.logger.info(f"{k} in the test set: {_cost[k]:.3f}")
            self.results['Initial test set'] = _cost

        try:
            # Print constraint satisfaction info about the predictions
            p = self.learner.predict(self.x)
            _cost = self.master.cst_info(self.x, p)
            for k in _cost.keys():
                self.logger.info(f"{k} in the training set, for the untrained model: {_cost[k]:.3f}")
            self.results['Untrained model, train set'] = _cost

            if self.x_test is not None:
                p = self.learner.predict(self.x_test)
                _cost = self.master.cst_info(self.x_test, p)
                for k in _cost.keys():
                    self.logger.info(f"{k} in the test set, for the untrained model: {_cost[k]:.3f}")
                    self.results['Untrained model'] = _cost
                self.results['Untrained model, test set'] = _cost

            # Initial accuracy
            _cost = self.master.score_info(self.y, self.learner.predict(self.x))
            for k in _cost.keys():
                self.logger.info(f"{k} on the train set, for the untrained model: {_cost[k]:.3f}")
            self.results['Untrained model, train set'] = _cost

            if self.x_test is not None:
                _cost = self.master.score_info(self.y_test, self.learner.predict(self.x_test))
                for k in _cost.keys():
                    self.logger.info(f"{k} on the test set, for the untrained model: {_cost[k]:.3f}")
                self.results['Untrained model, test set'] = _cost

        except sklearn.exceptions.NotFittedError as e:
            self.logger.info('Predictions for the untrained model are not available')

    def on_process_end(self):
        self.logger.info('>>> The MACS process is over')

    def on_pretraining_start(self):
        self.logger.info('--- Starting pretraining')
        self.ptstart_ = time.time()

    def on_pretraining_end(self):
        ttime = time.time() - self.ptstart_
        self.logger.info(f'Pretraining time: {ttime:.3f}')
        # Print constraint satisfaction info about the predictions
        p = self.learner.predict(self.x)
        train_cost = self.master.cst_info(self.x, p)
        for k in train_cost.keys():
            self.logger.info(f"{k} in the training set, for the pretrained model: {train_cost[k]:.3f}")
        self.results['Pretrained model, train set cost'] = train_cost

        if self.x_test is not None:
            p = self.learner.predict(self.x_test)
            test_cost = self.master.cst_info(self.x_test, p)
            for k in test_cost.keys():
                self.logger.info(f"{k} in the test set, for the pretrained model: {test_cost[k]:.3f}")
            self.results['Pretrained model, test set cost'] = test_cost

        # Accuracy
        train_score = self.master.score_info(self.y, self.learner.predict(self.x))
        for k in train_score.keys():
            self.logger.info(f"{k} on the train set, for the pretrained model: {train_score[k]:.3f}")
        self.results['Pretrained model, train set score'] = train_score

        if self.x_test is not None:
            test_score = self.master.score_info(self.y_test, self.learner.predict(self.x_test))
            for k in test_score.keys():
                self.logger.info(f"{k} on the test set, for the pretrained model: {test_score[k]:.3f}")
        self.results['Pretrained model, test set score'] = test_score

        self.logger.info('--- Pretraining done')

    def on_iteration_start(self, idx):
        self.logger.info(f'Iteration: {idx}')
        self.logger.info(f'=== Starting iteration')

    def on_iteration_end(self, idx):
        self.logger.info(f'=== Iteration {idx} is over')

    def on_training_start(self):
        self.logger.info('--- Starting to train the learner')
        self.trstart_ = time.time()

    def on_training_end(self, it=None):
        ttime = time.time() - self.trstart_
        self.logger.info(f'Training time: {ttime:.3f}')
        # Print constraint satisfaction info about the predictions
        p = self.learner.predict(self.x)
        train_cost = self.master.cst_info(self.x, p)
        for k in train_cost.keys():
            self.logger.info(f"{k} in the training set, for the model: {train_cost[k]:.3f}")
        self.results[f'Trained model it {it}, train set cost'] = train_cost

        if self.x_test is not None:
            p = self.learner.predict(self.x_test)
            test_cost = self.master.cst_info(self.x_test, p)
            for k in test_cost.keys():
                self.logger.info(f"{k} in the test set, for the model: {test_cost[k]:.3f}")
            self.results[f'Trained model it {it}, test set cost'] = test_cost

        # Accuracy
        train_score = self.master.score_info(self.y, self.learner.predict(self.x))
        for k in train_score.keys():
            self.logger.info(f"{k} on the train set, for the model: {train_score[k]:.3f}")
        self.results[f'Trained model it {it}, train set score'] = train_score

        if self.x_test is not None:
            test_score = self.master.score_info(self.y_test, self.learner.predict(self.x_test))
            for k in test_score.keys():
                self.logger.info(f"{k} on the test set, for the model: {test_score[k]:.3f}")
            self.results[f'Trained model it {it}, test set score'] = test_score

        self.logger.info('--- Training done')

    def on_adjustment_start(self):
        self.logger.info('--- Starting label adjustment')
        self.lastart_ = time.time()

    def on_adjustment_end(self, labels, it=None):
        ttime = time.time() - self.lastart_
        self.logger.info(f'Label adjustment time: {ttime:.3f}')
        # Print constraint satisfaction info about the adjusted labels
        _cost = self.master.cst_info(self.x, labels)
        for k in _cost.keys():
            self.logger.info(f"{k} in the training set, for the adjusted targets: {_cost[k]:.3f}")
        self.results[f'Adj. Targets it {it}, train set cost'] = _cost

        # Accuracy
        _cost = self.master.score_info(self.y, labels)
        for k in _cost.keys():
            self.logger.info(f"{k} on the train set, for the adjusted targets: {_cost[k]:.3f}")
        self.results[f'Adj. Targets it {it}, test set cost'] = _cost

        self.logger.info('--- Label adjustment done')

    @staticmethod
    def init(log_on_file=True, name='log', **kwargs):
        """
        Initialize logger
        """
        logger = logging.Logger(__name__)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        if log_on_file:
            file_handler = logging.FileHandler(f'{name}.log', 'w', 'utf-8')
            file_formatter = logging.Formatter(fmt='%(message)s',
                                               datefmt='%d/%m/%Y %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)

        for k, v in kwargs.items():
            logger.info(f'{k}: {v}')


class ResultsCollector:
    """
    Class that collects results for final analysis.
    """
    def __init__(self):
        self.results = dict()
        self.it = -1
        self.df = pd.DataFrame()
        self.next()

    def store(self, name, val):

        if name not in self.curdict.keys():
            self.curdict[name] = list()

        self.curdict[name].append(val)

    def next(self):
        self.it += 1
        self.results[self.it] = dict()

    def output(self):

        for it, val in self.results.items():
            _df = pd.DataFrame.from_dict(val, orient='columns')
            self.df = self.df.append(_df, ignore_index=True, sort=False)

        # out = pd.concat([self.df.mean(axis=0), self.df.std(axis=0)], axis=1).T
        # out.index = ['mean', 'std']

        val_list = list()
        col_list = list()
        for name, col in self.df.T.iterrows():
            val_list += [col.mean(), col.std()]
            col_list += [name, "std_%s" % name]

        out = pd.DataFrame(val_list).T
        out.columns = col_list

        print(out)
        return out

    @property
    def curdict(self):
        return self.results[self.it]

