import random
import numpy as np
import warnings
from six.moves import xrange
import tensorflow.compat.v1 as tf
import tensorflow_constrained_optimization as tfco
from source import macs, data_gen, utils
from sklearn.preprocessing import MinMaxScaler

from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score

tf.disable_eager_execution()

warnings.filterwarnings('ignore')
random.seed(42)


def _get_error_rate_and_constraints(preds, labels, didi, I):
    """
    Computes the error and constraint violations.
    """
    error = utils.mean_squared_error(preds, labels)
    ct_violation = utils.didi_r(preds, I) - 0.2 * didi
    return error, [ct_violation]


def _get_exp_error_rate_constraints(cand_dist, error_rates_vector, constraints_matrix):
    """Computes the expected error and fairness violations on a randomized solution."""
    expected_error_rate = np.dot(cand_dist, error_rates_vector)
    expected_constraints = np.matmul(cand_dist, constraints_matrix)
    return expected_error_rate, expected_constraints


# ===== CLASS TO BE USED BY MACS ===== #
class FairRegProblem(tfco.ConstrainedMinimizationProblem):
    """
    Define the classification problem and its constraints, without
    resorting to the pre-build RateMinimizationProblem (which does not
    work with multi-label classification).
    For now we just reproduce a naive result.
    """

    def __init__(self, labels, predictions, I_train, didi_tr):
        super(FairRegProblem, self).__init__()
        self.labels = labels
        self.predictions = predictions
        self.I_train = I_train

        self.perc_constraint_value = 0.2
        self.constraint_value = self.perc_constraint_value * didi_tr

    @property
    def num_constraints(self):
        return 1

    def objective(self):
        """
        The objective function of the constrained problem.
        """
        loss = tf.losses.mean_squared_error(self.labels, self.predictions)
        return loss

    def constraints(self):
        """
        The constraints to impose.
        """
        # Turn softmax output to categories.
        predictions = self.predictions

        # Set the constraint to zero.
        ct = list()

        for I in self.I_train:
            N = tf.reduce_sum(tf.cast(I >= 0, dtype=tf.float32))
            Np = tf.reduce_sum(I)
            a = (tf.reduce_sum(predictions) / N)
            b = (tf.reduce_sum(I * predictions) / Np)

            tmp = tf.cond(Np > 0, lambda: a - b, lambda: 0.0)
            ct.append(tf.abs(tmp))

        # ConstrainedMinimizationProblems must always provide their constraints in
        # the form (tensor <= 0).
        # return self.constraint - self.constraint_value
        return sum(ct) - self.constraint_value


class TFCOFairReg(macs.Learner):

    def __init__(self, input_dim, output_dim, I_train, didi_tr):
        super(TFCOFairReg).__init__()
        # tf.random.set_random_seed(123)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.I_train = I_train
        self.didi_tr = didi_tr
        self.protected_features = ['pfeat_%d' % k for k in I_train.keys()]
        self.fitted = False
        self.tpr_max_diff = 0

        # Initialize tf placeholders.
        self.features_placeholder = tf.placeholder(
            tf.float32, shape=(None, input_dim), name='features_placeholder')
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=(None, output_dim), name='labels_placeholder')
        self.protected_placeholders = [
            tf.placeholder(tf.float32, shape=(None, 1), name=attribute + "_placeholder") for attribute in
            self.protected_features]
        self.I = [
            tf.placeholder(tf.float32, shape=(None, 1), name='I_matrix_placeholder_' + str(mat)) for mat in
            self.I_train.keys()]

        # We use a network as a model, as in other examples.
        hidden_1 = tf.layers.dense(inputs=self.features_placeholder, units=32, activation='relu')
        hidden_2 = tf.layers.dense(inputs=hidden_1, units=32, activation='relu')
        self.predictions_tensor = tf.layers.dense(inputs=hidden_2, units=output_dim, activation=None)

        # Linear model.
        # self.predictions_tensor = tf.layers.dense(inputs=self.features_placeholder, units=output_dim, activation='linear')

        # Constrained minimization problem.
        self.mp = FairRegProblem(labels=self.labels_placeholder,
                                 predictions=self.predictions_tensor,
                                 I_train=self.I,
                                 didi_tr=self.didi_tr)

        # Start tf session.
        self.session = tf.Session()

    def build_train_op(self,
                       learning_rate=0.01):

        # opt = tfco.ProxyLagrangianOptimizerV1(tf.train.AdamOptimizer(learning_rate))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.opt = tfco.LagrangianOptimizerV1(optimizer)
        self.train_op = self.opt.minimize(self.mp)
        # self.train_op = optimizer.minimize(self.mp.objective())
        return self.train_op

    def _feed_dict_helper(self, x, y=None, I=None):

        feed_dict = {
            self.features_placeholder: x,
        }

        xp = x[:, -len(self.protected_features):]
        for i in range(len(self.protected_features)):
            feed_dict[self.protected_placeholders[i]] = xp[:, i].reshape(-1, 1)

        if y is not None:
            feed_dict[self.labels_placeholder] = y

        if I is not None:
            for i, mat in enumerate(I):
                feed_dict[self.I[i]] = mat.reshape(-1, 1)

        return feed_dict

    def _training_generator(self,
                            x,
                            y,
                            minibatch_size,
                            num_iterations_per_loop=1,
                            num_loops=1):
        num_rows = x.shape[0]
        minibatch_size = min(minibatch_size, num_rows)
        permutation = list(range(x.shape[0]))
        random.shuffle(permutation)
        # print(f"Fairness bound: {0.2 * self.didi_tr}")
        minibatch_start_index = 0
        for n in xrange(num_loops):
            for _ in xrange(num_iterations_per_loop):
                minibatch_indices = []
                while len(minibatch_indices) < minibatch_size:
                    minibatch_end_index = (
                            minibatch_start_index + minibatch_size - len(minibatch_indices))
                    if minibatch_end_index >= num_rows:
                        minibatch_indices += range(minibatch_start_index, num_rows)
                        minibatch_start_index = 0
                    else:
                        minibatch_indices += range(minibatch_start_index, minibatch_end_index)
                        minibatch_start_index = minibatch_end_index
                self.session.run(
                    self.train_op,
                    feed_dict=self._feed_dict_helper(
                        x[[permutation[ii] for ii in minibatch_indices]],
                        y[[permutation[ii] for ii in minibatch_indices]],
                        [I[[permutation[ii] for ii in minibatch_indices]] for I in self.I_train.values()]))
            # ct = self.session.run(
            #     self.mp.constraint,
            #     feed_dict=self._feed_dict_helper(
            #         x,
            #         y,
            #         [I for I in self.I_train.values()])
            # )
            # print(f"Loop {n}")
            # print("DIDItr: %.3f" % (0.2 * self.didi_tr))
            # print(f"TF Constraint value {ct}")
            slack = self.session.run(
                self.mp.constraints(),
                feed_dict=self._feed_dict_helper(
                    x,
                    y,
                    [I for I in self.I_train.values()])
            )
            # print(f"TF Slack value {slack}")

            p = self.session.run(
                self.predictions_tensor,
                feed_dict=self._feed_dict_helper(x)
            )

            preds = (1 + np.sign(p)) / 2
            perc_didi = utils.didi_r(preds, self.I_train) / self.didi_tr
            # print("Positive preds: %.0f / %.0f" % (sum(preds), len(preds)))
            # print("DIDI index: %.3f" % perc_didi)

            yield p

    def _training_helper(self,
                         x,
                         y,
                         minibatch_size,
                         num_iterations_per_loop=1,
                         num_loops=1):
        train_error_rate_vector = []
        train_constraints_matrix = []

        for train in self._training_generator(
                x, y, minibatch_size, num_iterations_per_loop,
                num_loops):

            train_error_rate, train_constraints = _get_error_rate_and_constraints(
                train, y.reshape(-1, 1), self.didi_tr, self.I_train)

            train_error_rate_vector.append(train_error_rate)
            train_constraints_matrix.append(train_constraints)

        return train_error_rate_vector, train_constraints_matrix

    def fit(self, x, y):

        self.build_train_op()
        self.session.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        res = self._training_helper(x, y, minibatch_size=1000, num_iterations_per_loop=326,
                                    num_loops=40)

        self.fitted = True
        return res

    def predict(self, x):

        if not self.fitted:
            raise NotFittedError

        p = self.session.run(
            self.predictions_tensor,
            feed_dict=self._feed_dict_helper(x))

        return p

    def _full_training(self,
                       xtr,
                       xts,
                       ytr,
                       minibatch_size,
                       num_iterations_per_loop=1,
                       num_loops=1):

        train_pred = []
        test_pred = []

        num_rows = xtr.shape[0]
        minibatch_size = min(minibatch_size, num_rows)
        permutation = list(range(num_rows))
        random.shuffle(permutation)

        minibatch_start_index = 0

        self.build_train_op()
        self.session.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

        for n in xrange(num_loops):
            for _ in xrange(num_iterations_per_loop):
                minibatch_indices = []
                while len(minibatch_indices) < minibatch_size:
                    minibatch_end_index = (
                            minibatch_start_index + minibatch_size - len(minibatch_indices))
                    if minibatch_end_index >= num_rows:
                        minibatch_indices += range(minibatch_start_index, num_rows)
                        minibatch_start_index = 0
                    else:
                        minibatch_indices += range(minibatch_start_index, minibatch_end_index)
                        minibatch_start_index = minibatch_end_index

                self.session.run(
                    self.train_op,
                    feed_dict=self._feed_dict_helper(
                        xtr[[permutation[ii] for ii in minibatch_indices]],
                        ytr[[permutation[ii] for ii in minibatch_indices]],
                        [I[[permutation[ii] for ii in minibatch_indices]] for I in self.I_train.values()]))

            # ct = self.session.run(
            #     self.mp.constraint,
            #     feed_dict=self._feed_dict_helper(
            #         x,
            #         y,
            #         [I for I in self.I_train.values()])
            # )
            # print(f"Loop {n}")
            # print("DIDItr: %.3f" % (0.2 * self.didi_tr))
            # print(f"TF Constraint value {ct}")
            slack = self.session.run(
                self.mp.constraints(),
                feed_dict=self._feed_dict_helper(
                    xtr,
                    ytr,
                    [I for I in self.I_train.values()])
            )
            # print(f"TF Slack value {slack}")

            # Multipliers.
            m = self.session.run(
                self.opt._formulation.state,
                feed_dict=self._feed_dict_helper(xtr)
            )
            # print(m)

            _train_pred = self.session.run(
                self.predictions_tensor,
                feed_dict=self._feed_dict_helper(xtr)
            )

            _test_pred = self.session.run(
                self.predictions_tensor,
                feed_dict=self._feed_dict_helper(xts)
            )

            train_pred.append(_train_pred)
            test_pred.append(_test_pred)

        return train_pred, test_pred


def test():

    # Data with our preprocessing routines.
    from sklearn.preprocessing import MinMaxScaler

    xnp, xp, y = data_gen.load_crime()
    scl = MinMaxScaler()
    train_pts = int(0.8 * len(xnp))
    xnptr = scl.fit_transform(xnp[:train_pts])
    xnpts = scl.transform(xnp[train_pts:])
    xptr = xp[:train_pts]
    xpts = xp[train_pts:]
    ytr = y[:train_pts]
    yts = y[train_pts:]

    # Add protected features.
    xtr = np.hstack([xnptr, xptr])
    xts = np.hstack([xnpts, xpts])

    scl = MinMaxScaler()
    ytr = scl.fit_transform(ytr)
    yts = scl.transform(yts)

    I_train = utils.compute_indicator_matrix_r(xptr)
    I_test = utils.compute_indicator_matrix_r(xpts)
    didi_tr = utils.didi_r(ytr, I_train)
    didi_ts = utils.didi_r(yts, I_test)

    tfco_model = TFCOFairReg(input_dim=xtr.shape[1], output_dim=1,
                             I_train=I_train, didi_tr=didi_tr)

    # Fitting.
    # train_errors, train_violations = tfco_model.fit(xtr, ytr)
    # train_errors, train_violations = np.array(train_errors), np.array(train_violations)

    # test_errors, test_violations = tfco_model.predict_err(x_ts.values, y_ts.values)
    # test_preds = tfco_model.predict(xts)
    # test_errors, test_violations = _get_error_rate_and_constraints(
    #     test_preds, yts, didi_ts, I_test)

    minibatch_size = 200
    iterations_per_loop = 200
    loops = 80

    train_pred, test_pred = tfco_model._full_training(xtr, xts, ytr,
                                                      minibatch_size, iterations_per_loop, loops)

    train_errors = []
    train_violations = []
    train_didi = []
    train_r2 = []

    for p in train_pred:
        err, viol = _get_error_rate_and_constraints(p, ytr.reshape(-1, 1), didi_tr, I_train)
        r2 = r2_score(ytr, p)
        didi = utils.didi_r(p, I_train) / didi_tr
        train_errors.append(err)
        train_violations.append(viol)
        train_didi.append(didi)
        train_r2.append(r2)

    test_errors = []
    test_violations = []
    test_didi = []
    test_r2 = []

    for p in test_pred:
        err, viol = _get_error_rate_and_constraints(p, yts.reshape(-1, 1), didi_ts, I_test)
        r2 = r2_score(yts, p)
        didi = utils.didi_r(p, I_test) / didi_ts
        test_errors.append(err)
        test_violations.append(viol)
        test_didi.append(didi)
        test_r2.append(r2)

    train_violations = np.array(train_violations)
    # print("DIDI train", didi_tr)
    # print("DIDI test", didi_ts)
    # print("Train Error", train_errors[-1])
    # print("Train Violation", max(train_violations[-1]))
    print("Train R2", train_r2[-1])
    print("Train DIDI", train_didi[-1])

    # print("Test Error", test_errors[-1])
    # print("Test Violation", max(test_violations[-1]))
    print("Train R2", test_r2[-1])
    print("Train DIDI", test_didi[-1])

    print("Improving using Best Iterate instead of Last Iterate.")
    #
    # As discussed in [[CotterEtAl18b]](https://arxiv.org/abs/1809.04198), the last iterate may not be the best choice
    # and suggests a simple heuristic to choose the best iterate out of the ones found after each epoch.
    # The heuristic proceeds by ranking each of the solutions based on accuracy and fairness separately with respect to
    # the training data. Any solutions which satisfy the constraints are equally ranked top in terms fairness.
    # Each solution thus has two ranks. Then, the chosen solution is the one with the smallest maximum of the two ranks.
    # We see that this improves the fairness and can find a better accuracy / fairness trade-off on the training data.
    #
    # This solution can be calculated using find_best_candidate_index given the list of training errors and violations
    # associated with each of the epochs.

    best_cand_index = tfco.find_best_candidate_index(train_errors, train_violations)

    # print("Train Error", train_errors[best_cand_index])
    # print("Train Violation", max(train_violations[best_cand_index]))
    print("Train R2", train_r2[best_cand_index])
    print("Train DIDI", train_didi[best_cand_index])

    # print("Test Error", test_errors[best_cand_index])
    # print("Test Violation", max(test_violations[best_cand_index]))
    print("Test R2", test_r2[best_cand_index])
    print("Test DIDI", test_didi[best_cand_index])

    print("m-stochastic solution.")
    # [[CoJiSr19]](https://arxiv.org/abs/1804.06500) presents a method which shrinks down the T-stochastic solution down
    # to one that is supported on at most (m+1) points where m is the number of constraints and is guaranteed to be at
    # least as good as the T-stochastic solution.
    # Here we see that indeed there is benefit in performing the shrinking.
    #
    # This solution can be computed using find_best_candidate_distribution by passing in the training errors and
    # violations found at each epoch and returns the weight of each constituent. We see that indeed, it is sparse.

    cand_dist = tfco.find_best_candidate_distribution(train_errors, train_violations)
    print(cand_dist)

    # m_stoch_error_train, m_stoch_violations_train = _get_exp_error_rate_constraints(cand_dist, train_errors,
    #                                                                                 train_violations)
    # m_stoch_error_test, m_stoch_violations_test = _get_exp_error_rate_constraints(cand_dist, test_errors,
    #                                                                               test_violations)

    m_stoch_train_r2 = np.dot(cand_dist, train_r2)
    m_stoch_train_didi = np.dot(cand_dist, train_didi)
    m_stoch_test_r2 = np.dot(cand_dist, test_r2)
    m_stoch_test_didi = np.dot(cand_dist, test_didi)

    print("Train R2", m_stoch_train_r2)
    print("Train DIDI", m_stoch_train_didi)
    print("Test R2", m_stoch_test_r2)
    print("Test DIDI", m_stoch_test_didi)


def cross_val():

    # Data with our preprocessing routines.

    xnp, xp, y = data_gen.load_crime()

    results = {
        'Last_iterate_train_acc': [],
        'Last_iterate_test_acc': [],
        'Last_iterate_train_ct': [],
        'Last_iterate_test_ct': [],
        'Best_iterate_train_acc': [],
        'Best_iterate_test_acc': [],
        'Best_iterate_train_ct': [],
        'Best_iterate_test_ct': [],
        'Stoch_iterate_train_acc': [],
        'Stoch_iterate_test_acc': [],
        'Stoch_iterate_train_ct': [],
        'Stoch_iterate_test_ct': [],
    }
    nfolds = 5
    fsize = int(np.ceil(len(xnp) / nfolds))
    for fidx in range(nfolds):
        print(f'\n### Processing fold {fidx}')
        # Build a full index set
        idx = np.arange(len(xnp))
        # Separate index sets
        tridx = np.hstack((idx[:fidx*fsize], idx[(fidx+1)*fsize:]))
        tsidx = idx[fidx*fsize:(fidx+1)*fsize]
        # Separate training and test data
        xtr = xnp[tridx]
        xptr = xp[tridx]
        ytr = y[tridx]
        xts = xnp[tsidx]
        xpts = xp[tsidx]
        yts = y[tsidx]

        # Standardize train set.
        scl = MinMaxScaler()
        xtr = scl.fit_transform(xtr)
        xts = scl.transform(xts)

        # Add protected features.
        xtr = np.hstack([xtr, xptr])
        xts = np.hstack([xts, xpts])

        scl = MinMaxScaler()
        ytr = scl.fit_transform(ytr)
        yts = scl.transform(yts)

        print("Computing indicator matrices.")
        I_train = utils.compute_indicator_matrix_r(xptr)
        I_test = utils.compute_indicator_matrix_r(xpts)
        didi_tr = utils.didi_r(ytr, I_train)
        didi_ts = utils.didi_r(yts, I_test)

        tfco_model = TFCOFairReg(input_dim=xtr.shape[1], output_dim=1,
                                 I_train=I_train, didi_tr=didi_tr)

        # Fitting.
        # train_errors, train_violations = tfco_model.fit(xtr, ytr)
        # train_errors, train_violations = np.array(train_errors), np.array(train_violations)

        # test_errors, test_violations = tfco_model.predict_err(x_ts.values, y_ts.values)
        # test_preds = tfco_model.predict(xts)
        # test_errors, test_violations = _get_error_rate_and_constraints(
        #     test_preds, yts, didi_ts, I_test)

        minibatch_size = 200
        iterations_per_loop = 200
        loops = 100

        train_pred, test_pred = tfco_model._full_training(xtr, xts, ytr,
                                                          minibatch_size, iterations_per_loop, loops)

        train_errors = []
        train_violations = []
        train_didi = []
        train_r2 = []

        for p in train_pred:
            err, viol = _get_error_rate_and_constraints(p, ytr.reshape(-1, 1), didi_tr, I_train)
            r2 = r2_score(ytr, p)
            didi = utils.didi_r(p, I_train) / didi_tr
            train_errors.append(err)
            train_violations.append(viol)
            train_didi.append(didi)
            train_r2.append(r2)

        test_errors = []
        test_violations = []
        test_didi = []
        test_r2 = []

        for p in test_pred:
            err, viol = _get_error_rate_and_constraints(p, yts.reshape(-1, 1), didi_ts, I_test)
            r2 = r2_score(yts, p)
            didi = utils.didi_r(p, I_test) / didi_ts
            test_errors.append(err)
            test_violations.append(viol)
            test_didi.append(didi)
            test_r2.append(r2)

        train_violations = np.array(train_violations)
        # print("DIDI train", didi_tr)
        # print("DIDI test", didi_ts)
        # print("Train Error", train_errors[-1])
        # print("Train Violation", max(train_violations[-1]))
        print("Train R2", train_r2[-1])
        print("Train DIDI", train_didi[-1])

        # print("Test Error", test_errors[-1])
        # print("Test Violation", max(test_violations[-1]))
        print("Train R2", test_r2[-1])
        print("Train DIDI", test_didi[-1])

        print("Improving using Best Iterate instead of Last Iterate.")
        #
        # As discussed in [[CotterEtAl18b]](https://arxiv.org/abs/1809.04198), the last iterate may not be the best choice
        # and suggests a simple heuristic to choose the best iterate out of the ones found after each epoch.
        # The heuristic proceeds by ranking each of the solutions based on accuracy and fairness separately with respect to
        # the training data. Any solutions which satisfy the constraints are equally ranked top in terms fairness.
        # Each solution thus has two ranks. Then, the chosen solution is the one with the smallest maximum of the two ranks.
        # We see that this improves the fairness and can find a better accuracy / fairness trade-off on the training data.
        #
        # This solution can be calculated using find_best_candidate_index given the list of training errors and violations
        # associated with each of the epochs.

        best_cand_index = tfco.find_best_candidate_index(train_errors, train_violations)

        # print("Train Error", train_errors[best_cand_index])
        # print("Train Violation", max(train_violations[best_cand_index]))
        print("Train R2", train_r2[best_cand_index])
        print("Train DIDI", train_didi[best_cand_index])

        # print("Test Error", test_errors[best_cand_index])
        # print("Test Violation", max(test_violations[best_cand_index]))
        print("Test R2", test_r2[best_cand_index])
        print("Test DIDI", test_didi[best_cand_index])

        print("m-stochastic solution.")
        # [[CoJiSr19]](https://arxiv.org/abs/1804.06500) presents a method which shrinks down the T-stochastic solution down
        # to one that is supported on at most (m+1) points where m is the number of constraints and is guaranteed to be at
        # least as good as the T-stochastic solution.
        # Here we see that indeed there is benefit in performing the shrinking.
        #
        # This solution can be computed using find_best_candidate_distribution by passing in the training errors and
        # violations found at each epoch and returns the weight of each constituent. We see that indeed, it is sparse.

        cand_dist = tfco.find_best_candidate_distribution(train_errors, train_violations)
        print(cand_dist)

        # m_stoch_error_train, m_stoch_violations_train = _get_exp_error_rate_constraints(cand_dist, train_errors,
        #                                                                                 train_violations)
        # m_stoch_error_test, m_stoch_violations_test = _get_exp_error_rate_constraints(cand_dist, test_errors,
        #                                                                               test_violations)

        m_stoch_train_r2 = np.dot(cand_dist, train_r2)
        m_stoch_train_didi = np.dot(cand_dist, train_didi)
        m_stoch_test_r2 = np.dot(cand_dist, test_r2)
        m_stoch_test_didi = np.dot(cand_dist, test_didi)

        print("Train R2", m_stoch_train_r2)
        print("Train DIDI", m_stoch_train_didi)
        print("Test R2", m_stoch_test_r2)
        print("Test DIDI", m_stoch_test_didi)

        results['Last_iterate_train_acc'].append(train_r2[-1])
        results['Last_iterate_test_acc'].append(test_r2[-1])
        results['Last_iterate_train_ct'].append(train_didi[-1])
        results['Last_iterate_test_ct'].append(test_didi[-1])

        results['Best_iterate_train_acc'].append(train_r2[best_cand_index])
        results['Best_iterate_test_acc'].append(test_r2[best_cand_index])
        results['Best_iterate_train_ct'].append(train_didi[best_cand_index])
        results['Best_iterate_test_ct'].append(test_didi[best_cand_index])

        results['Stoch_iterate_train_acc'].append(m_stoch_train_r2)
        results['Stoch_iterate_test_acc'].append(m_stoch_test_r2)
        results['Stoch_iterate_train_ct'].append(m_stoch_train_didi)
        results['Stoch_iterate_test_ct'].append(m_stoch_test_didi)


    print("\nTOTAL RESULTS")
    for k, val in results.items():
        print(k, np.mean(val), np.std(val))


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TFCO library')

    # Configure options

    parser.add_argument('--check', action='store_true',
                        help='Performe consistency tests.')

    args = parser.parse_args()
    if args.check:
        raise NotImplementedError
    else:
        # test()
        cross_val()
