"""
Module containing all the masters optimization model used.
"""

import numpy as np
from docplex.mp.model import Model as CPModel
from docplex.mp.model import DOcplexException
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from source import utils

_CPLEX_TIME_LIMIT = 30


class BalancedCountsMaster:
    """docstring for BalancedCountsMaster"""

    def __init__(self, nclasses):
        super(BalancedCountsMaster, self).__init__()
        self.nclasses = nclasses

    def adjust_targets(self, y, p, alpha, beta, use_prob=False):
        assert(alpha == 0 or p is not None)
        print("Master problem")
        # Obtain the number of samples
        nsamples = len(y)
        # Detect the number of classes
        classes, cnt = np.unique(y, return_counts=True)
        nclasses = len(classes)
        # Compute counts
        # maxc = int(np.ceil(nsamples / nclasses))
        maxc = int(np.ceil((1.05 * nsamples) / nclasses))
        # Determine feasibility
        _, pcnts = np.unique(p, return_counts=True)
        feasible = np.all(pcnts <= maxc)
        # Build a model
        mod = CPModel('Class balancer')

        # Set a time limit.
        mod.parameters.timelimit = _CPLEX_TIME_LIMIT

        # Build the decision variables
        vy = mod.binary_var_matrix(keys1=nsamples,
                                   keys2=nclasses, name='y')
        # Constrain the class counts
        for c in classes:
            xpr = mod.sum([vy[i, c] for i in range(nsamples)])
            mod.add_constraint(xpr <= maxc)
        # A class should be assigned to each example
        for i in range(nsamples):
            xpr = mod.sum(vy[i, c] for c in range(nclasses))
            mod.add_constraint(xpr == 1)
        # Define the loss w.r.t. the true labels
        p_loss = (1 / nsamples) * mod.sum([(1-vy[i, p[i]]) for i in range(nsamples)])
        y_loss = (1 / nsamples) * mod.sum([(1-vy[i, y[i]]) for i in range(nsamples)])
        if feasible and beta >= 0:
            # Search in a Ball
            # 1/alpha determines the allowed number of flips from the original solution.
            # dataset_scale = 0.1
            # mod.add(p_loss <= dataset_scale/alpha)
            mod.add(p_loss <= beta)
            # Minimize distance w.r.t. the targets
            mod.minimize(y_loss)
        else:
            # Project (with tie breaking)
            mod.minimize(y_loss + (1.0 / alpha) * p_loss)

            # 231020: Ball search.
            # First I compute the minimum range that assures feasibility and then impose
            # it as a costraint.
            # mod2 = mod.clone("Radius model")
            # mod2.minimize(nsamples * p_loss)
            # mod2.solve()
            # r = mod2.objective_value
            # print("Objective value (radius): %.2f" % r)

            # mod.add(p_loss <= (1.05 * r))
            # mod.minimize(y_loss)

        # Solve the problem
        sol = mod.solve()
        # Parse the found solution
        self._check_solution(mod)
        # Obtain the adjusted labels
        ya = [sum(c * sol.get_value(vy[i, c])
            for c in range(nclasses)) for i in range(nsamples)]
        y_opt = np.array([int(v) for v in ya])

        print("Total flips: %d" % np.sum(np.abs(y_opt - p)))

        return y_opt

    def cst_info(self, x, y):

        cnts = np.array([np.sum(y == c) for c in range(self.nclasses)])
        s = ', '.join([f'{v}:{n}' for v, n in zip(range(self.nclasses), cnts)])
        # print(f'Value counts in {dset_label} = {s}')
        print(f'Value counts = {s}')
        std = np.std(cnts / np.sum(cnts))
        # ent = stats.entropy(cnts/np.sum(cnts))
        # print(f'Std. Dev. of class frequencies in {dset_label}: {std:.3f}')

        cost = {
            'Std. Dev. of class frequencies': std
        }
        return cost

    def score_info(self, y, yhat):
        acc = accuracy_score(y, yhat)
        score = {
            'Accuracy': acc,
        }
        return score

    @staticmethod
    def _check_solution(mod):

        try:
            mod.check_has_solution()
            print("Solver status: " + mod.solve_details.status)
            print("MIP Gap: {:.3f} %".format(100 * mod.solve_details.mip_relative_gap))
        except DOcplexException as err:
            # self.logger.error("Model infeasible!")
            mod.export_as_lp(path='./', basename='infeasible')
            raise err


class FairnessRegMaster:

    def __init__(self, I_train, I_test, didi_tr, didi_ts):
        super(FairnessRegMaster, self).__init__()
        self.I_test = I_test
        self.I_train = I_train
        self.didi_tr = didi_tr
        self.didi_ts = didi_ts

        self.perc_constraint_value = 0.2
        self.constraint_value = self.perc_constraint_value * didi_tr

    def cst_info(self, x, y):
        """
        Print information about the cost (satisfaction) associated to the inputs.
        """
        # Infer train /test set from the input arrays.
        I = None
        d = None
        n_points = len(x)
        if n_points == len(self.I_train[0]):
            I = self.I_train
            d = self.didi_tr
        elif n_points == len(self.I_test[0]):
            I = self.I_test
            d = self.didi_ts
        else:
            raise ValueError("Cannot infer indicator matrix from input data. Input array has "
                             "shape %d, with matrices having shape %d and %d" % (n_points,
                                                                                 len(self.I_train[0]),
                                                                                 len(self.I_test[0])))

        perc_didi = utils.didi_r(y, I) / d
        cost = {
            'DIDI perc. index': perc_didi
        }
        return cost

    def score_info(self, y, yhat):
        """
        Print information about the score of the inputs.
        """
        mse = mean_squared_error(y, yhat)
        r2 = r2_score(y, yhat)

        score = {
            'MSE': mse,
            'R2': r2,
        }
        return score

    def adjust_targets(self, y, p, alpha, beta, use_prob):
        """
        Solve the optimization model that returns the optimal prediction that satisfy the constraints.
        """
        assert (alpha == 0 or p is not None)
        # self.logger.debug("Setting up Opt Model")

        # Input adjusting.
        y = y.reshape(-1)
        p = p.reshape(-1)

        # Determine feasibility
        _feasible = (utils.didi_r(p, self.I_train) <= self.constraint_value)

        # Model declaration.
        mod = CPModel('Fairness Reg Problem')

        # Set a time limit.
        mod.parameters.timelimit = _CPLEX_TIME_LIMIT

        # Variable declaration.
        n_points = len(y)
        idx_var = [i for i in range(n_points)]
        x = mod.continuous_var_list(keys=idx_var, lb=0.0, ub=1.0, name='y')

        # Fairness constraint: instead of adding a penalization term in the objective function - as done by
        # Phebe et al - I impose the fairness term to stay below a certain threshold.
        constraint = .0
        abs_val = mod.continuous_var_list(keys=self.I_train.keys())
        for key, val in self.I_train.items():
            Np = np.sum(val)
            if Np > 0:
                tmp = (1.0 / n_points) * mod.sum(x) - \
                      (1.0 / Np) * mod.sum([val[j] * x[j] for j in idx_var])
                # Linearization of the absolute value.
                mod.add_constraint(abs_val[key] >= tmp)
                mod.add_constraint(abs_val[key] >= -tmp)

        constraint += mod.sum(abs_val)
        mod.add_constraint(constraint <= self.constraint_value, ctname='fairness_cnst')

        # Objective Function.
        y_loss = (1.0 / n_points) * mod.sum([(y[i] - x[i]) * (y[i] - x[i]) for i in idx_var])
        p_loss = (1.0 / n_points) * mod.sum([(p[i] - x[i]) * (p[i] - x[i]) for i in idx_var])

        if _feasible and beta >= 0:
            # Constrain search on a ball.
            mod.add(p_loss <= beta)
            mod.minimize(y_loss)
        else:
            # Adds a regularization term to make sure the new targets are not too far from the actual
            # network's output.
            mod.minimize(y_loss + (1.0 / alpha) * p_loss)

            # 231020: Ball search.
            # First I compute the minimum range that assures feasibility and then impose
            # it as a costraint.
            # mod2 = mod.clone("Radius model")
            # mod2.minimize(n_points * p_loss)
            # mod2.solve()
            # r = mod2.objective_value
            # print("Objective value (radius): %.2f" % r)

            # mod.add(p_loss <= (1.05 * r))
            # mod.minimize(y_loss)

        # Problem solving.
        # self.logger.info("Solving Opt Model...")
        mod.solve()

        # Check solution.
        self._check_solution(mod)

        # Obtain the adjusted targets.
        y_opt = np.array([x[i].solution_value for i in range(n_points)])

        return y_opt

    @staticmethod
    def _check_solution(mod):

        try:
            mod.check_has_solution()
            print("Solver status: " + mod.solve_details.status)
            print("MIP Gap: {:.3f} %".format(100 * mod.solve_details.mip_relative_gap))
        except DOcplexException as err:
            # self.logger.error("Model infeasible!")
            mod.export_as_lp(path='./', basename='infeasible')
            raise err


class FairnessClsMaster:

    def __init__(self, I_train, I_test, didi_tr, didi_ts):
        super(FairnessClsMaster, self).__init__()
        self.I_test = I_test
        self.I_train = I_train
        self.didi_tr = didi_tr
        self.didi_ts = didi_ts

        self.perc_constraint_value = 0.2
        self.constraint_value = self.perc_constraint_value * didi_tr

    def cst_info(self, x, y):
        """
        Print information about the cost (satisfaction) associated to the inputs.
        """
        # Infer train /test set from the input arrays.
        I = None
        d = None
        n_points = len(x)
        if n_points == len(self.I_train[0]):
            I = self.I_train
            d = self.didi_tr
        elif n_points == len(self.I_test[0]):
            I = self.I_test
            d = self.didi_ts
        else:
            raise ValueError("Cannot infer indicator matrix from input data. Input array has "
                             "shape %d, with matrices having shape %d and %d" % (n_points,
                                                                                 len(self.I_train[0]),
                                                                                 len(self.I_test[0])))

        perc_didi = utils.didi_c(y, I) / d
        cost = {
            'DIDI perc. index': perc_didi
        }

        return cost

    def score_info(self, y, yhat):
        """
        Print information about the score of the inputs.
        """
        acc = accuracy_score(y, yhat)
        score = {
            'Accuracy': acc
        }

        return score

    def adjust_targets(self, y, p, alpha, beta, use_prob=False):
        """
        Solve the optimization model that returns the optimal prediction that satisfy the constraints.
        """
        assert (alpha == 0 or p is not None)
        # self.logger.debug("Setting up Opt Model")

        if use_prob:
            prob = p.copy()
            # Output clipping to avoid infinities.
            prob = np.clip(prob, a_min=.01, a_max=.99)
            p = np.argmax(prob, axis=1)

        # Input adjusting.
        y = y.reshape(-1)
        p = p.reshape(-1)

        # Determine feasibility
        _feasible = (utils.didi_c(p, self.I_train) <= self.constraint_value)
        print(f'Current solution is feasible: {_feasible}')

        # Model declaration.
        mod = CPModel('Fairness Cls Problem')

        # Set a time limit (seconds).
        mod.parameters.timelimit = _CPLEX_TIME_LIMIT

        # Variable declaration.
        n_points = len(y)
        idx_var = [i for i in range(n_points)]
        x = mod.binary_var_list(keys=idx_var, name='y')

        # Fairness constraint: instead of adding a penalization term in the objective function - as done by
        # Phebe et al - I impose the fairness term to stay below a certain threshold.
        # self.logger.debug("...constraints declaration")
        constraint = .0
        abs_val = mod.continuous_var_list(keys=self.I_train.keys())
        for key, I in self.I_train.items():
            # print(key, i, var_i)
            Np = np.sum(I)
            if Np > 0:
                tmp = 2 * (mod.sum(x) / n_points -
                           mod.sum([I[j] * x[j] for j in idx_var]) / Np)

                mod.add_constraint(abs_val[key] >= tmp)
                mod.add_constraint(abs_val[key] >= -tmp)

        constraint += mod.sum(abs_val)
        mod.add_constraint(constraint <= self.constraint_value, ctname='fairness_cnst')

        # Objective Function.
        y_loss = (1.0 / n_points) * mod.sum([y[i] * (1 - x[i]) + (1 - y[i]) * x[i] for i in idx_var])
        if use_prob:
            p_loss = - (1.0 / n_points) * mod.sum([x[i] * np.log(prob[i][1]) + (1-x[i]) * np.log(prob[i][0])
                                                   for i in idx_var])
        else:
            p_loss = (1.0 / n_points) * mod.sum([p[i] * (1 - x[i]) + (1 - p[i]) * x[i] for i in idx_var])

        if _feasible and beta >= 0:
            # Search in a Ball
            # 1/alpha determines the allowed number of flips from the original solution.
            # dataset_scale = 0.01 * n_points
            mod.add(p_loss <= beta)
            # Minimize distance w.r.t. the targets
            mod.minimize(y_loss)
        else:
            # Project (with tie breaking)
            mod.minimize(y_loss + (1.0 / alpha) * p_loss)

            # 231020: Ball search.
            # First I compute the minimum range that assures feasibility and then impose
            # it as a costraint.
            # mod2 = mod.clone("Radius model")
            # mod2.minimize(n_points * p_loss)
            # mod2.solve()
            # self._check_solution(mod2)
            # r = mod2.objective_value
            # print("Objective value (radius): %.2f" % r)

            # mod.add(p_loss <= (1.05 * r))
            # mod.minimize(y_loss)

        # Problem solving.
        mod.solve()

        # Check solution.
        self._check_solution(mod)

        # Obtain the adjusted targets.
        y_opt = np.array([int(x[i].solution_value) for i in range(n_points)])

        print("Total flips: %d" % np.sum(np.abs(y_opt-p)))

        return y_opt

    @staticmethod
    def _check_solution(mod):

        try:
            mod.check_has_solution()
            print("Solver status: " + mod.solve_details.status)
            print("MIP Gap: {:.3f} %".format(100 * mod.solve_details.mip_relative_gap))
        except DOcplexException as err:
            # self.logger.error("Model infeasible!")
            mod.export_as_lp(path='./', basename='infeasible')
            raise err


# class NegativePatternMaster(object):
#     """docstring for NegativePatternMaster"""

#     def __init__(self):
#         super(NegativePatternMaster, self).__init__()

#     def get_idxset_(self, x):
#         mask = np.mean(x, axis=1) > 0
#         idxset = np.arange(nsamples)[mask]
#         return idxset

#     def adjust_labels(self, x, y, p, alpha, use_prob=False):
#         assert(alpha == 0 or p is not None)
#         # Obtain the number of samples and classes
#         nsamples =  len(y)
#         classes, cnt = np.unique(y, return_counts=True)
#         nclasses = len(classes)
#         # Detect the examples whose class should be changed
#         idxset = self.get_idxset_(x)
#         print(idxset)
#         sys.exit()
#         # Buid a model
#         mod = CPModel('Pair differences')

#         # Build the decision variables
#         vy = mod.binary_var_matrix(keys1=nsamples,
#                 keys2=nclasses, name='y')
#         # Add pair constraints
#         for k in range(len(f.pairs)):
#             for c in range(nclasses):
#                 xpr = mod.sum([vy[i, c] for i in self.pairs[k, :]])
#                 mod.add_constraint(xpr <= 1)
#         # A class should be assigned to each example
#         for i in range(nsamples):
#             xpr = mod.sum(vy[i, c] for c in range(nclasses))
#             mod.add_constraint(xpr == 1)
#         # Define the loss w.r.t. the true labels
#         y_loss = mod.sum([(1-vy[i, y[i]]) for i in range(nsamples)])
#         # Define the loss w.r.t. the predicted labels
#         if alpha > 0:
#             if not use_prob:
#                 p_loss = mod.sum([(1-vy[i, p[i]]) for i in range(nsamples)])
#             else:
#                 raise ValueError('Probabilistic predictions are not yet supported')
#             mod.minimize(y_loss + alpha * p_loss)
#         else:
#             mod.minimize(y_loss)
#         # Solve the problem
#         sol = mod.solve()
#         # Parse the found solution
#         if sol is None:
#             # Infeasible/unbuonded... Something went wrong
#             return sol
#         else:
#             # Obtain the adjusted labels
#             ya = [sum(c * sol.get_value(vy[i,c])
#                 for c in range(nclasses)) for i in range(nsamples)]
#             return np.array([int(v) for v in ya])


#     def cst_info(self, x, y, nclasses, dset_label=''):
#         idxset = self.get_idxset_(x)
#         cnt = np.sum()

#         vcnt = np.sum(y[self.pairs[:, 0]] == y[self.pairs[:, 1]])
#         print(f'Violated pair constraints in {dset_label} = {vcnt}')
#         vfrac = vcnt / len(self.pairs)
#         print(f'Violated constraint fraction in {dset_label} = {vfrac:.3f}')
