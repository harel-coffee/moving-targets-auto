import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from source.data_gen import Dataset
from source.data_gen import BALANCE_DATASET
from source import macs, tfco_reg, tfco_cls, utils
from source.masters import BalancedCountsMaster, FairnessRegMaster, FairnessClsMaster
from source.models import regressors as rgs
from source.models import classifiers as cls
from source.logger import CustomLogger
from source.wandb_logger import WandBLogger


class Validation:

    def __init__(self, dataset, nfolds, mtype, ltype, iterations,
                 alpha, beta, initial_step, use_prob, scaler='minmax', test_size=0.2):
        """
        Evaluate the performance of the model on a rolling test set, with n_periods sets made of
        test_samples test points.
        Tne other parameters are used to build the model.
        """
        self.dataset = dataset
        self.nfolds = nfolds
        self.mtype = mtype
        self.ltype = ltype
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.initial_step = initial_step
        self.use_prob = use_prob
        self.test_size = test_size

        if scaler == 'standard':
            self.scaler = StandardScaler()
        elif scaler == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f'Unknown scaler "{scaler}"')

        self.results = dict()

        # Load all the data.
        D = Dataset(self.dataset, self.test_size)
        self.xnp_tr = D.xnp_tr
        self.xp_tr = D.xp_tr
        self.xnp_ts = D.xnp_ts
        self.xp_ts = D.xp_ts
        self.y_tr = D.y_tr
        self.y_ts = D.y_ts

        # Log instance info
        self.logger = CustomLogger.init(
            dataset=self.dataset,
            nfolds=self.nfolds,
            mtype=self.mtype,
            ltype=self.ltype,
            iterations=self.iterations,
            alpha=self.alpha,
            beta=self.beta,
            initial_step=self.initial_step,
            use_prob=use_prob,
            test_size=self.test_size
        )

        self.learner = None
        self.master = None

    def get_train_val_index(self, fold):

        # Build a full index set
        idx = np.arange(len(self.xnp_tr))
        fsize = int(np.ceil(len(self.xnp_tr) / self.nfolds))

        # Separate index sets
        train_index = np.hstack((idx[:fold * fsize], idx[(fold + 1) * fsize:]))
        test_index = idx[fold * fsize:(fold + 1) * fsize]

        return train_index, test_index

    def validate(self):

        # Load data.
        # self.load_data()

        for ii in range(self.nfolds):

            # TRAIN TEST SPLIT
            if self.dataset in BALANCE_DATASET:
                train_idx, test_idx = self.get_train_val_index(ii)
                xnp_train, y_train = self.xnp_tr[train_idx], self.y_tr[train_idx]
                xnp_test, y_test = self.xnp_tr[test_idx], self.y_tr[test_idx]

                # STANDARDIZATION.
                # Standardize train set.
                x_train = self.scaler.fit_transform(xnp_train)
                x_test = self.scaler.transform(xnp_test)

                # y_train = self.scaler.fit_transform(y_train)
                # y_test = self.scaler.transform(y_test)

            else:
                train_idx, test_idx = self.get_train_val_index(ii)
                xnp_train, xp_train, y_train = self.xnp_tr[train_idx], self.xp_tr[train_idx], self.y_tr[train_idx]
                xnp_test, xp_test, y_test = self.xnp_tr[test_idx], self.xp_tr[test_idx], self.y_tr[test_idx]

                # STANDARDIZATION.
                xnp_train = self.scaler.fit_transform(xnp_train)
                xnp_test = self.scaler.transform(xnp_test)

                # Add protected features.
                x_train = np.hstack([xnp_train, xp_train])
                x_test = np.hstack([xnp_test, xp_test])

                y_train = self.scaler.fit_transform(y_train)
                y_test = self.scaler.transform(y_test)

            if self.dataset in BALANCE_DATASET:
                # Data shapes.
                input_dim = x_train.shape[1]
                output_dim = len(np.unique(y_train))

                # Build the master
                if self.mtype == 'balance':
                    nclasses = len(np.unique(y_train))
                    self.master = BalancedCountsMaster(nclasses=nclasses)
                else:
                    raise ValueError(f'Unknown master type "{self.mtype}"')

                # Start the main process
                if self.ltype == 'cvx':
                    self.learner = cls.BalanceMultiLogRegressor(self.alpha)

                elif self.ltype == 'sbrnn':
                    self.learner = cls.SBRNN(input_dim, output_dim, self.alpha)

                elif self.ltype == 'lbrf':
                    self.learner = cls.LowBiasRandomForestLearner(input_dim, output_dim)

                elif self.ltype == 'lr':
                    self.learner = cls.LogisticRegressionLearner(input_dim, output_dim)

                elif self.ltype == 'rf':
                    self.learner = cls.RandomForestLearner(input_dim, output_dim)

                elif self.ltype == 'nn':
                    self.learner = cls.NeuralNetworkLearner(input_dim, output_dim)

                else:
                    raise ValueError(f'Unknown learner type "{self.ltype}"')

            elif self.dataset == 'adult':
                print("Computing indicator matrices.")
                I_train = utils.compute_indicator_matrix_c(xp_train)
                I_test = utils.compute_indicator_matrix_c(xp_test)
                didi_tr = utils.didi_c(y_train, I_train)
                didi_ts = utils.didi_c(y_test, I_test)

                # Build the master
                if self.mtype == 'fairness':
                    self.master = FairnessClsMaster(I_train, I_test, didi_tr, didi_ts)
                else:
                    raise ValueError(f'Unknown master type "{self.mtype}"')

                input_dim = x_train.shape[1]
                output_dim = len(np.unique(y_train))

                # Start the main process
                if self.ltype == 'cvx':
                    self.learner = cls.FairBinLogRegressor(self.alpha, I_train)

                elif self.ltype == 'cnd':
                    # Kamiran and Calders method.
                    # learner = cls.CND(xnptr, xptr, ytr)
                    raise NotImplementedError

                elif self.ltype == 'tfco':
                    input_dim = x_train.shape[1]
                    output_dim = 1
                    self.learner = tfco_cls.TFCOFairCls(input_dim, output_dim, I_train, didi_tr)

                elif self.ltype == 'lbrf':
                    self.learner = cls.LowBiasRandomForestLearner(input_dim, output_dim)

                elif self.ltype == 'lr':
                    self.learner = cls.LogisticRegressionLearner(input_dim, output_dim)

                elif self.ltype == 'rf':
                    self.learner = cls.RandomForestLearner(input_dim, output_dim)

                elif self.ltype == 'nn':
                    self.learner = cls.NeuralNetworkLearner(input_dim, output_dim)

                else:
                    raise ValueError(f'Unknown learner type "{self.ltype}"')

            elif self.dataset == 'crime':
                print("Computing indicator matrices.")
                I_train = utils.compute_indicator_matrix_r(xp_train)
                I_test = utils.compute_indicator_matrix_r(xp_test)
                didi_tr = utils.didi_r(y_train, I_train)
                didi_ts = utils.didi_r(y_test, I_test)

                # Build the master
                if self.mtype == 'fairness':
                    self.master = FairnessRegMaster(I_train, I_test, didi_tr, didi_ts)
                else:
                    raise ValueError(f'Unknown master type "{self.mtype}"')

                # Build the learner.
                if self.ltype == 'cvx':
                    self.learner = rgs.FairRegressor(self.alpha, I_train)

                elif self.ltype == 'tfco':
                    input_dim = x_train.shape[1]
                    output_dim = 1
                    self.learner = tfco_reg.TFCOFairReg(input_dim, output_dim, I_train, didi_tr)

                elif self.ltype == 'lbrf':
                    self.learner = rgs.LowBiasRandomForestLearner()

                elif self.ltype == 'lr':
                    self.learner = rgs.LRegressor()

                elif self.ltype == 'gb':
                    self.learner = rgs.GBTree()

                elif self.ltype == 'nn':
                    self.learner = rgs.Net((x_train.shape[1],), 1)

                else:
                    raise ValueError(f'Unknown learner type "{self.ltype}"')

            # Start the MACS process
            # logger = CustomLogger(self.learner, self.master, x_train, y_train, nfold=ii, x_test=x_test, y_test=y_test)
            p = dict(fold=ii, alpha=self.alpha, beta=self.beta, init=self.initial_step, use_prob=self.use_prob)
            self.logger = WandBLogger(self.learner, self.master, x_train, y_train, x_test, y_test, p, f'{self.dataset}')
            mp = macs.MACS(self.learner, self.master, self.logger)
            mp.fit(x_train, y_train, self.iterations, self.alpha, self.beta, self.initial_step, use_prob=self.use_prob)
            self.results[f'fold_{ii}'] = self.logger.results

    def test(self):
        # Evaluation of the model on the test set.

        # TRAIN TEST SPLIT
        if self.dataset in BALANCE_DATASET:
            xnp_train, y_train = self.xnp_tr, self.y_tr
            xnp_test, y_test = self.xnp_ts, self.y_ts

            # STANDARDIZATION.
            # Standardize train set.
            x_train = self.scaler.fit_transform(xnp_train)
            x_test = self.scaler.transform(xnp_test)

            # y_train = self.scaler.fit_transform(y_train)
            # y_test = self.scaler.transform(y_test)

        else:
            xnp_train, xp_train, y_train = self.xnp_tr, self.xp_tr, self.y_tr
            xnp_test, xp_test, y_test = self.xnp_ts, self.xp_ts, self.y_ts

            # STANDARDIZATION.
            xnp_train = self.scaler.fit_transform(xnp_train)
            xnp_test = self.scaler.transform(xnp_test)

            # Add protected features.
            x_train = np.hstack([xnp_train, xp_train])
            x_test = np.hstack([xnp_test, xp_test])

            y_train = self.scaler.fit_transform(y_train)
            y_test = self.scaler.transform(y_test)

        if self.dataset in BALANCE_DATASET:
            # Data shapes.
            input_dim = xnp_train.shape[1]
            output_dim = len(np.unique(y_train))

            # Build the master
            if self.mtype == 'balance':
                nclasses = len(np.unique(y_train))
                self.master = BalancedCountsMaster(nclasses=nclasses)
            else:
                raise ValueError(f'Unknown master type "{self.mtype}"')

            # Start the main process
            if self.ltype == 'cvx':
                self.learner = cls.BalanceMultiLogRegressor(self.alpha)

            elif self.ltype == 'sbrnn':
                self.learner = cls.SBRNN(input_dim, output_dim, self.alpha)

            elif self.ltype == 'lbrf':
                self.learner = cls.LowBiasRandomForestLearner(input_dim, output_dim)

            elif self.ltype == 'lr':
                self.learner = cls.LogisticRegressionLearner(input_dim, output_dim)

            elif self.ltype == 'rf':
                self.learner = cls.RandomForestLearner(input_dim, output_dim)

            elif self.ltype == 'nn':
                self.learner = cls.NeuralNetworkLearner(input_dim, output_dim)

            else:
                raise ValueError(f'Unknown learner type "{self.ltype}"')

        elif self.dataset == 'adult':
            print("Computing indicator matrices.")
            I_train = utils.compute_indicator_matrix_c(xp_train)
            I_test = utils.compute_indicator_matrix_c(xp_test)
            didi_tr = utils.didi_c(y_train, I_train)
            didi_ts = utils.didi_c(y_test, I_test)

            # Build the master
            if self.mtype == 'fairness':
                self.master = FairnessClsMaster(I_train, I_test, didi_tr, didi_ts)
            else:
                raise ValueError(f'Unknown master type "{self.mtype}"')

            input_dim = x_train.shape[1]
            output_dim = len(np.unique(y_train))

            # Start the main process
            if self.ltype == 'cvx':
                self.learner = cls.FairBinLogRegressor(self.alpha, I_train)

            elif self.ltype == 'cnd':
                # Kamiran and Calders method.
                # learner = cls.CND(xnptr, xptr, ytr)
                raise NotImplementedError

            elif self.ltype == 'tfco':
                input_dim = x_train.shape[1]
                output_dim = 1
                self.learner = tfco_cls.self.learner = tfco_cls.TFCOFairCls(input_dim, output_dim, I_train, didi_tr)

            elif self.ltype == 'lbrf':
                self.learner = cls.LowBiasRandomForestLearner(input_dim, output_dim)

            elif self.ltype == 'lr':
                self.learner = cls.LogisticRegressionLearner(input_dim, output_dim)

            elif self.ltype == 'rf':
                self.learner = cls.RandomForestLearner(input_dim, output_dim)

            elif self.ltype == 'nn':
                self.learner = cls.NeuralNetworkLearner(input_dim, output_dim)

            else:
                raise ValueError(f'Unknown learner type "{self.ltype}"')

        elif self.dataset == 'crime':
            print("Computing indicator matrices.")
            I_train = utils.compute_indicator_matrix_r(xp_train)
            I_test = utils.compute_indicator_matrix_r(xp_test)
            didi_tr = utils.didi_r(y_train, I_train)
            didi_ts = utils.didi_r(y_test, I_test)

            # Build the master
            if self.mtype == 'fairness':
                self.master = FairnessRegMaster(I_train, I_test, didi_tr, didi_ts)
            else:
                raise ValueError(f'Unknown master type "{self.mtype}"')

            # Build the learner.
            if self.ltype == 'cvx':
                self.learner = rgs.FairRegressor(self.alpha, I_train)

            elif self.ltype == 'tfco':
                input_dim = x_train.shape[1]
                output_dim = 1
                self.learner = tfco_reg.TFCOFairReg(input_dim, output_dim, I_train, didi_tr)

            elif self.ltype == 'lbrf':
                self.learner = rgs.LowBiasRandomForestLearner()

            elif self.ltype == 'lr':
                self.learner = rgs.LRegressor()

            elif self.ltype == 'gb':
                self.learner = rgs.GBTree()

            elif self.ltype == 'nn':
                self.learner = rgs.Net((x_train.shape[1],), 1)

            else:
                raise ValueError(f'Unknown learner type "{self.ltype}"')

        # Start the MACS process
        # logger = CustomLogger(self.learner, self.master, x_train, y_train, nfold=99, x_test=x_test, y_test=y_test)
        p = dict(fold=ii, alpha=self.alpha, beta=self.beta, init=self.initial_step, use_prob=self.use_prob)
        self.logger = WandBLogger(self.learner, self.master, x_train, y_train, x_test, y_test, p, f'{self.dataset}')
        mp = macs.MACS(self.learner, self.master, self.logger)
        mp.fit(x_train, y_train, self.iterations, self.alpha, self.beta, self.initial_step, self.use_prob)
        self.results['Test'] = self.logger.results

    def collect_results(self):
        # Collect the results stored in the logger during the n-fold validation procedure
        # and compute statistical indicators on them (mean and std. dev.).

        self.logger.logger.info("\n### Collecting results")
        tot_results = dict()
        for fold, res in self.results.items():
            for key, val in res.items():
                for x, y in val.items():
                    name = key + "_" + x
                    if name not in tot_results.keys():
                        tot_results[name] = list()
                    tot_results[name].append(y)

        # Print results.
        for key, val in tot_results.items():
            self.logger.logger.info("%s: %.4f (%.4f)" % (key, np.mean(val), np.std(val)))

        # Empty the collector.
        self.results = dict()
