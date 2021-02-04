import random
import numpy as np
import warnings
from six.moves import xrange
import tensorflow.compat.v1 as tf
from source import macs, data_gen
import tensorflow_constrained_optimization as tfco
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
# Data with our preprocessing routines.
from sklearn.preprocessing import StandardScaler

tf.disable_eager_execution()

warnings.filterwarnings('ignore')


def _get_error_rate_and_constraints(preds, labels):
    """
    Computes the error and constraint violations.
    """
    Nc = len(np.unique(labels))
    N = len(labels)
    maxc = int(np.ceil(1.05 * N / Nc))
    error = 1 - accuracy_score(preds, labels)
    ct_violation = []
    for i in range(Nc):
        _nc = sum(preds == i)
        ct_violation.append(_nc - maxc)
    return error, ct_violation


# ===== CLASS TO BE USED BY MACS ===== #
class FairBalProblem(tfco.ConstrainedMinimizationProblem):
    """
    Define the classification problem and its constraints, without
    resorting to the pre-build RateMinimizationProblem (which does not
    work with multi-label classification).
    For now we just reproduce a naive result.
    """

    def __init__(self, labels, predictions, num_classes, num_samples):
        super(FairBalProblem, self).__init__()
        self.labels = labels
        self.predictions = predictions
        self.num_classes = num_classes
        self.num_samples = num_samples

    @property
    def num_constraints(self):
        return self.num_classes
        # return 1

    def objective(self):
        """
        The objective function of the constrained problem.
        """
        loss = tf.keras.losses.categorical_crossentropy(self.labels, self.predictions)
        # loss = tf.losses.mean_squared_error(self.labels, self.predictions)
        return tf.reduce_sum(loss)

    def constraints(self):
        """
        The constraints to impose.
        """
        # Turn softmax output to categories.
        predictions = tf.cast(tf.argmax(self.predictions, axis=1), dtype=tf.float32)

        # Set the constraint to zero.
        constraint_list = []
        eps = .05
        maxc = int(np.ceil((1 + eps) * self.num_samples / self.num_classes))
        counts = tf.math.bincount(tf.cast(predictions, dtype=tf.int32), minlength=self.num_classes, maxlength=self.num_classes)
        # counts = tf.Print(counts, [counts])

        for c in range(self.num_classes):
            arr = tf.cast(predictions >= c, dtype=tf.float32) * tf.cast(predictions < c+1, dtype=tf.float32)
            Nc = tf.reduce_sum(arr) / self.num_samples
            # Nc = tf.reduce_sum(tf.cast(predictions == c, dtype=tf.float32))
            # Nc = tf.reduce_sum(tf.cast(predictions > c, dtype=tf.float32))
            # Nc = tf.cast(counts[c], dtype=tf.float32)
            ct = Nc - maxc
            # ct = Nc
            # ct = tf.reduce_sum(self.predictions[c]) - maxc
            constraint_list.append(ct)

        # ConstrainedMinimizationProblems must always provide their constraints in
        # the form (tensor <= 0).
        # return tf.stack(tf.reduce_sum(self.predictions, axis=0))
        return tf.stack(constraint_list)


class TFCOFairBal(macs.Learner):

    def __init__(self, input_dim, output_dim, num_classes, num_samples):
        super(TFCOFairBal).__init__()
        # tf.random.set_random_seed(123)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.fitted = False
        self.tpr_max_diff = 0

        # Initialize tf placeholders.
        self.features_placeholder = tf.placeholder(
            tf.float32, shape=(None, input_dim), name='features_placeholder')
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=(None, output_dim), name='labels_placeholder')

        # We use a network as a model, as in other examples.
        hidden_1 = tf.layers.dense(inputs=self.features_placeholder, units=32, activation='relu')
        hidden_2 = tf.layers.dense(inputs=hidden_1, units=32, activation='relu')
        self.predictions_tensor = tf.layers.dense(inputs=hidden_2, units=output_dim, activation='softmax')

        # Constrained minimization problem.
        self.mp = FairBalProblem(labels=self.labels_placeholder,
                                 predictions=self.predictions_tensor,
                                 num_classes=self.num_classes,
                                 num_samples=self.num_samples)

        # Start tf session.
        self.session = tf.Session()

    def build_train_op(self,
                       learning_rate=0.1):

        # opt = tfco.ProxyLagrangianOptimizerV1(tf.train.AdamOptimizer(learning_rate))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.opt = tfco.LagrangianOptimizerV1(optimizer)
        self.train_op = self.opt.minimize(self.mp)
        # self.train_op = optimizer.minimize(self.mp.objective())
        return self.train_op

    def build_train_op_tfco(self,
                            learning_rate=0.1):
        ctx = tfco.multiclass_rate_context(self.num_classes,
                                           self.predictions_tensor,
                                           self.labels_placeholder)
        # positive_slice = ctx.subset(self.labels_placeholder > 0)
        # overall_tpr = tfco.positive_prediction_rate(positive_slice)
        constraints = []
        for c in range(self.num_classes):
            pos_rate = tfco.positive_prediction_rate(ctx, c)
            constraints.append(pos_rate <= (1.05 / self.num_classes))
        mp = tfco.RateMinimizationProblem(tfco.error_rate(ctx), constraints)
        self.opt = tfco.ProxyLagrangianOptimizerV1(tf.train.AdamOptimizer(learning_rate))
        self.train_op = self.opt.minimize(mp)
        return self.train_op

    def _feed_dict_helper(self, x, y=None, I=None):

        feed_dict = {
            self.features_placeholder: x,
        }

        if y is not None:
            yc = tf.keras.utils.to_categorical(
                y, num_classes=self.num_classes, dtype='float32'
            )
            feed_dict[self.labels_placeholder] = yc

        return feed_dict

    def _training_generator(self,
                            x,
                            y,
                            minibatch_size,
                            num_iterations_per_loop=1,
                            num_loops=1):
        random.seed(31337)
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
                        y[[permutation[ii] for ii in minibatch_indices]]))
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
                    y)
            )
            # print(f"TF Slack value {slack}")

            p = self.session.run(
                self.predictions_tensor,
                feed_dict=self._feed_dict_helper(x)
            )

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

            train_c = tf.argmax(train, axis=1)
            train_error_rate, train_constraints = _get_error_rate_and_constraints(
                train_c, y
            )

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

        return tf.argmax(p, axis=1)

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

        # self.build_train_op()
        self.build_train_op_tfco()
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
                        ytr[[permutation[ii] for ii in minibatch_indices]]))

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
                    ytr)
            )
            # print(f"TF Slack value {slack}")

            # Multipliers.
            m = self.session.run(
                self.opt._formulation.state,
                feed_dict=self._feed_dict_helper(xtr)
            )
            # print(f"Multpliers {m}")

            _train_pred = self.session.run(
                self.predictions_tensor,
                feed_dict=self._feed_dict_helper(xtr)
            )

            _test_pred = self.session.run(
                self.predictions_tensor,
                feed_dict=self._feed_dict_helper(xts)
            )

            train_pred.append(np.argmax(_train_pred, axis=1))
            test_pred.append(np.argmax(_test_pred, axis=1))

        return train_pred, test_pred


def test():

    # Data with our preprocessing routines.
    from sklearn.preprocessing import MinMaxScaler

    x, y = data_gen.load_shuttle_data()
    scl = MinMaxScaler()
    train_pts = int(0.8 * len(x))
    xtr = scl.fit_transform(x[:train_pts])
    xts = scl.transform(x[train_pts:])
    ytr = y[:train_pts]
    yts = y[train_pts:]

    num_samples = len(xtr)
    num_classes = len(np.unique(ytr))
    tfco_model = TFCOFairBal(input_dim=xtr.shape[1], output_dim=num_classes,
                             num_classes=num_classes, num_samples=num_samples)

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
    train_acc = []
    train_std = []

    for p in train_pred:
        err, viol = _get_error_rate_and_constraints(p, ytr)
        acc = accuracy_score(ytr, p)
        cnts = np.array([np.sum(p == c) for c in range(num_classes)])
        std = np.std(cnts / np.sum(cnts))
        train_errors.append(err)
        train_violations.append(viol)
        train_acc.append(acc)
        train_std.append(std)

    test_errors = []
    test_violations = []
    test_acc = []
    test_std = []

    for p in test_pred:
        err, viol = _get_error_rate_and_constraints(p, yts)
        acc = accuracy_score(yts, p)
        cnts = np.array([np.sum(p == c) for c in range(num_classes)])
        std = np.std(cnts / np.sum(cnts))
        test_errors.append(err)
        test_violations.append(viol)
        test_acc.append(acc)
        test_std.append(std)

    train_violations = np.array(train_violations)
    print("Train Error", train_errors[-1])
    print("Train Violation", max(train_violations[-1]))
    print("Train Acc.", train_acc[-1])
    print("Train Std.", train_std[-1])

    print("Test Error", test_errors[-1])
    print("Test Violation", max(test_violations[-1]))
    print("Test Acc,", test_acc[-1])
    print("Test Std.", test_std[-1])

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
    print("Train Acc.", train_acc[best_cand_index])
    print("Train Std.", train_std[best_cand_index])

    # print("Test Error", test_errors[best_cand_index])
    # print("Test Violation", max(test_violations[best_cand_index]))
    print("Test Acc.", test_acc[best_cand_index])
    print("Test Std.", test_std[best_cand_index])

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

    m_stoch_train_r2 = np.dot(cand_dist, train_acc)
    m_stoch_train_didi = np.dot(cand_dist, train_std)
    m_stoch_test_r2 = np.dot(cand_dist, test_acc)
    m_stoch_test_didi = np.dot(cand_dist, test_std)

    print("Train Acc.", m_stoch_train_r2)
    print("Train Std.", m_stoch_train_didi)
    print("Test Acc.", m_stoch_test_r2)
    print("Test Std.", m_stoch_test_didi)


def cross_val():

    x, y = data_gen.load_dota_data()
    print("Dota")
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

    # Process the folds
    nfolds = 5
    fsize = int(np.ceil(len(x) / nfolds))
    for fidx in range(nfolds):
        print(f'\n### Processing fold {fidx}')
        # Build a full index set
        idx = np.arange(len(x))
        # Separate index sets
        tridx = np.hstack((idx[:fidx * fsize], idx[(fidx + 1) * fsize:]))
        tsidx = idx[fidx * fsize:(fidx + 1) * fsize]
        # Separate training and test data
        xtr = x[tridx]
        ytr = y[tridx]
        xts = x[tsidx]
        yts = y[tsidx]
        # Standardize
        # for cidx in range(x.shape[1]):
        scl = StandardScaler()
        xtr = scl.fit_transform(xtr)
        xts = scl.transform(xts)

        num_samples = len(xtr)
        num_classes = len(np.unique(ytr))
        tfco_model = TFCOFairBal(input_dim=xtr.shape[1], output_dim=num_classes,
                                 num_classes=num_classes, num_samples=num_samples)

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
        train_acc = []
        train_std = []

        for p in train_pred:
            err, viol = _get_error_rate_and_constraints(p, ytr)
            acc = accuracy_score(ytr, p)
            cnts = np.array([np.sum(p == c) for c in range(num_classes)])
            std = np.std(cnts / np.sum(cnts))
            train_errors.append(err)
            train_violations.append(viol)
            train_acc.append(acc)
            train_std.append(std)

        test_errors = []
        test_violations = []
        test_acc = []
        test_std = []

        for p in test_pred:
            err, viol = _get_error_rate_and_constraints(p, yts)
            acc = accuracy_score(yts, p)
            cnts = np.array([np.sum(p == c) for c in range(num_classes)])
            std = np.std(cnts / np.sum(cnts))
            test_errors.append(err)
            test_violations.append(viol)
            test_acc.append(acc)
            test_std.append(std)

        train_violations = np.array(train_violations)
        print("Train Error", train_errors[-1])
        print("Train Violation", max(train_violations[-1]))
        print("Train Acc.", train_acc[-1])
        print("Train Std.", train_std[-1])

        print("Test Error", test_errors[-1])
        print("Test Violation", max(test_violations[-1]))
        print("Test Acc,", test_acc[-1])
        print("Test Std.", test_std[-1])

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
        print("Train Acc.", train_acc[best_cand_index])
        print("Train Std.", train_std[best_cand_index])

        # print("Test Error", test_errors[best_cand_index])
        # print("Test Violation", max(test_violations[best_cand_index]))
        print("Test Acc.", test_acc[best_cand_index])
        print("Test Std.", test_std[best_cand_index])

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

        m_stoch_train_r2 = np.dot(cand_dist, train_acc)
        m_stoch_train_didi = np.dot(cand_dist, train_std)
        m_stoch_test_r2 = np.dot(cand_dist, test_acc)
        m_stoch_test_didi = np.dot(cand_dist, test_std)

        print("Train Acc.", m_stoch_train_r2)
        print("Train Std.", m_stoch_train_didi)
        print("Test Acc.", m_stoch_test_r2)
        print("Test Std.", m_stoch_test_didi)

        results['Last_iterate_train_acc'].append(train_acc[-1])
        results['Last_iterate_test_acc'].append(test_acc[-1])
        results['Last_iterate_train_ct'].append(train_std[-1])
        results['Last_iterate_test_ct'].append(test_std[-1])

        results['Best_iterate_train_acc'].append(train_acc[best_cand_index])
        results['Best_iterate_test_acc'].append(test_acc[best_cand_index])
        results['Best_iterate_train_ct'].append(train_std[best_cand_index])
        results['Best_iterate_test_ct'].append(test_std[best_cand_index])

        results['Stoch_iterate_train_acc'].append(m_stoch_train_r2)
        results['Stoch_iterate_test_acc'].append(m_stoch_test_r2)
        results['Stoch_iterate_train_ct'].append(m_stoch_train_didi)
        results['Stoch_iterate_test_ct'].append(m_stoch_test_didi)

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
