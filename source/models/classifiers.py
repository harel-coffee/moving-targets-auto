"""
Module containing all the regressive models considered.
"""

import numpy as np

from source.macs import Learner

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import GaussianNB

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense

import cvxpy as cp


# ======================================================================
# Logistic Regressors with regularization.
# ======================================================================


class FairBinLogRegressor():
    """
    """

    def __init__(self, alpha, I_train, gamma=0):

        self.alpha = alpha
        self.gamma = gamma
        self.I_train = I_train
        super(FairBinLogRegressor, self).__init__()
        self.w = None

    def log_likelihood(self, x, y, w):

        y = np.array(y).flatten()
        log_likelihood = cp.sum(
            cp.multiply(y, x @ w) - cp.logistic(x @ w)
        )
        return log_likelihood

    def fit(self, x, y):

        n_points = x.shape[0]
        n_feat = x.shape[1]
        w = cp.Variable(shape=n_feat)
        constraints = []

        fairness = 0
        for key, Ip in self.I_train.items():
            Np = np.sum(Ip)
            if Np >= 0:
                tmp = (x @ w) / n_points - cp.multiply(Ip, (x @ w)) / Np
                fairness += 2 * cp.abs(cp.sum(tmp))

        loglike = - self.log_likelihood(x, y, w) / n_points
        fair_reg = self.alpha * fairness
        w_reg = self.gamma * cp.sum_squares(w)

        obj_fct = loglike + fair_reg + w_reg

        prob = cp.Problem(cp.Minimize(obj_fct), constraints)
        prob.solve()
        # prob.solve(verbose=False, solver=cp.SCS)
        # prob.solve(verbose=True, solver=cp.ECOS, feastol=1e-5, abstol=1e-5)

        print("Value of log likelihood: %.3f" % loglike.value)
        print("Value of fairness regularization: %.3f" % fairness.value)

        self.w = w.value

    def fit_MIP(self, x, y):

        n_points = x.shape[0]
        n_feat = x.shape[1]
        w = cp.Variable(shape=n_feat)
        # Additional variables representing the actual predictions.
        yhat = cp.Variable(shape=n_points, boolean=True)
        bigM = 1e3
        constraints = [
            (yhat-1) * bigM <= x @ w,
            yhat * bigM >= x @ w,
        ]

        fairness = 0
        for key, Ip in self.I_train.items():
            Np = np.sum(Ip)
            if Np >= 0:
                # tmp = 2 * (cp.sum(x @ w) / n_points -
                #            cp.sum(cp.multiply(Ip, (x @ w))) / Np)

                tmp = 2 * (cp.sum(yhat) / n_points -
                           cp.sum(cp.multiply(Ip, yhat)) / Np)
                fairness += cp.abs(tmp)

        loglike = -self.log_likelihood(x, y, w) / n_points
        fair_reg = self.alpha * fairness
        w_reg = self.gamma * cp.sum_squares(w)

        obj_fct = loglike + fair_reg + w_reg
        prob = cp.Problem(cp.Minimize(obj_fct), constraints)
        prob.solve()
        # prob.solve(verbose=False, solver=cp.SCS)
        # prob.solve(verbose=True, solver=cp.ECOS, feastol=1e-5, abstol=1e-5)
        print("Value of log likelihood: %.3f" % loglike.value)
        print("Value of fairness regularization: %.3f" % fairness.value)

        self.w = w.value

    def predict(self, x):

        if self.w is None:
            raise NotFittedError

        p = x.dot(self.w)
        return (1 + np.sign(p)) / 2


class BalanceMultiLogRegressor:

    def __init__(self, alpha):
        super(BalanceMultiLogRegressor, self).__init__()
        self.weights = None
        self.alpha = alpha

    def fit(self, x, y):
        # Detect the number of samples and classes
        nsamples = x.shape[0]
        ncols = x.shape[1]
        classes, cnt = np.unique(y, return_counts=True)
        nclasses = len(classes)
        # Convert classes to a categorical format
        yc = keras.utils.to_categorical(y, num_classes=nclasses)

        # Build a disciplined convex programming model
        w = cp.Variable(shape=(ncols, nclasses))
        constraints = []
        # Additional variables representing the actual predictions.
        log_reg = x @ w
        Z = [cp.log_sum_exp(log_reg[i]) for i in range(nsamples)]
        log_likelihood = cp.sum(
            cp.sum([cp.multiply(yc[:, c], log_reg[:, c]) for c in range(nclasses)])) - cp.sum(Z)

        reg = cp.max(cp.sum(log_reg), axis=0)

        # Start the training process
        obj_func = - log_likelihood / nsamples + self.alpha * reg
        problem = cp.Problem(cp.Minimize(obj_func), constraints)
        problem.solve()

        self.weights = w.value

    def fit_OLD(self, x, y):
        # Detect the number of samples and classes
        nsamples = x.shape[0]
        ncols = x.shape[1]
        classes, cnt = np.unique(y, return_counts=True)
        nclasses = len(classes)
        # Convert classes to a categorical format
        yc = keras.utils.to_categorical(y, num_classes=nclasses)

        # Build a disciplined convex programming model
        w = cp.Variable(shape=(ncols, nclasses))
        # Additional variables representing the actual predictions.
        yhat = cp.Variable(shape=(nsamples, nclasses), boolean=True)
        bigM = 1e3
        constraints = [
            cp.sum(yhat, axis=1) == 1,  # only one class per sample.
        ]
        constraints += [
            x @ w[:, i] - x @ w[:, i+1] <= bigM * (yhat[:, i] - yhat[:, i+1]) for i in range(nclasses - 1)
        ]
        log_reg = x @ w
        # out_xpr = [cp.exp(log_out_xpr[c]) for c in range(nclasses)]
        Z = [cp.log_sum_exp(log_reg[i]) for i in range(nsamples)]
        # log_likelihood = cp.sum(
        #     cp.sum([cp.multiply(yc[:, c], log_out_xpr[c])
        #             for c in range(nclasses)]) - Z
        # )
        log_likelihood = cp.sum(
            cp.sum([cp.multiply(yc[:, c], log_reg[:, c]) for c in range(nclasses)])) - cp.sum(Z)

        reg = 0
        # Compute counts
        maxc = int(np.ceil(nsamples / nclasses))
        for c in classes:
            reg += cp.square(maxc - cp.sum(yhat[c]))

        # Start the training process
        obj_func = - log_likelihood / nsamples + self.alpha * reg
        problem = cp.Problem(cp.Minimize(obj_func), constraints)
        problem.solve()

        # for c in range(nclasses):
        #     wgt[c] = cp.Variable(ncols)
        #     # xpr = cp.sum(cp.multiply(y, x @ wgt) - cp.logistic(x @ wgt))
        #     log_out_xpr[c] = x @ wgt[c]
        #     out_xpr[c] = cp.exp(x @ wgt[c])
        #     if c == 0: log_likelihood = xpr
        #     else: log_likelihood += xpr
        # problem = cp.Problem(cp.Maximize(log_likelihood/nsamples))
        # # Start the training process
        # problem.solve()

        # Store the weights
        # print(wgt.value)
        self.weights = w.value

    def predict(self, x):

        if self.weights is None:
            raise sklearn.exceptions.NotFittedError

        log_reg = np.exp(np.matmul(x, self.weights))
        pred = log_reg / log_reg.sum(axis=1).reshape(-1, 1)
        cat_pred = np.argmax(pred, axis=1).reshape(-1)
        # print(pred)
        # print(cat_pred)
        return cat_pred

    def predict_proba(self, x):
        raise NotImplementedError


# ======================================================================
# Kamiran and Calders approach
# ======================================================================


class CND(object):
    """
    Method described by Kamiran and Calders.
    The input (biased) data are "adjusted" via an heuristic procedure and
    then fed to the classifier, which is the Naive Bayes method.

    The class is to be used in the following way: the master will call just
    the preprocessing step and no iterations. In this way, targets will not be
    "moved" by the MT algorithm, but only in the preprocessing step.
    """
    def __init__(self, xnp, xp, y):
        self.xnp = xnp
        self.xp = xp
        self.y = y
        self.model = GaussianNB()

    def fit(self, x, y):

        y_cnd = self._adjust_labels()
        self.model.fit(x, y_cnd)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def _adjust_labels(self):

        # Get the promotion candidates and their index in y array.
        promotion_mask = (self.xp == 1) & (self.y == 0)
        promotion_list = np.where(promotion_mask)[0]

        # Get the demotion candidates and their index in y array.
        demotion_mask = (self.xp == 0) & (self.y == 1)
        demotion_list = np.where(demotion_mask)[0]

        ranker = GaussianNB()
        ranker.fit(self.xnp, self.y)
        yprob = ranker.predict_proba(self.xnp)

        print(yprob.shape)
        print(promotion_mask.shape)

        # Use the ranker to sort promotion and demotion list order
        # according to its prediciton probability.
        promotion_proba = yprob[promotion_mask, -1]
        demotion_proba = yprob[demotion_mask][:, -1]
        promotion_list = promotion_list[np.argsort(promotion_proba)]
        demotion_list = demotion_list[np.argsort(demotion_proba)]

        # Compute how many labels are to be switched.
        M = 2
        M = np.sum(self.xp) * np.sum((1 - self.xp) * self.y) - \
            np.sum(1 - self.xp) * np.sum(self.xp * self.y)
        M = M / len(self.xp)

        if M < 0:
            raise ValueError("Negative number of switch required! :c")

        # Adjust targets.
        y_cnd = self.y.copy()
        for i in range(M):
            promotion_idx = promotion_list[i]
            demotion_idx = demotion_list[i]
            y_cnd[promotion_idx] = 1
            y_cnd[demotion_idx] = 0

        return y_cnd


# ======================================================================
# Semantic Based Regularization NN
# ======================================================================


class SBRNN(object):
    """docstring for SBRNN"""

    def __init__(self, input_dim, output_dim, alpha, epochs=1600):
        super(SBRNN, self).__init__()
        # Store the weight for the regularization terms
        self.alpha = alpha
        # Build a basic NN model
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=input_dim, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(output_dim, activation='softmax'))
        self.epochs = epochs
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fitted = False

    def fit(self, x, y):
        # Custom loss function
        batch_size = 2048
        maxc = np.ceil(batch_size / self.output_dim)

        def sbr_loss(yt, yp):
            cnts = K.sum(yp, axis=0)
            cmax = K.max(cnts)
            # nsamples = K.shape(yt)[0]
            # nsamples = K.cast(nsamples, cmax.dtype)

            # return K.categorical_crossentropy(yt, yp) + \
            #         self.alpha * K.sum(K.maximum(0.0, cnts-maxc))
            return K.categorical_crossentropy(yt, yp) + self.alpha * cmax

        # One hot encoding
        yc = keras.utils.to_categorical(y, num_classes=self.output_dim)
        # Compile
        self.model.compile(optimizer='rmsprop',
                           loss=sbr_loss,
                           metrics=['accuracy'])
        # Train
        self.model.fit(x, yc, epochs=self.epochs,
                       batch_size=batch_size, verbose=0)
        self.fitted = True

    def predict(self, x):
        if not self.fitted:
            raise NotFittedError
        p = self.model.predict(x)
        return np.argmax(p, axis=1)

    def predict_proba(self, x):
        return self.model.predict(x)


# ======================================================================
# Learners
# ======================================================================

class NeuralNetworkLearner(Learner):
    """docstring for LogisticRegressionLearner"""

    def __init__(self, input_dim, output_dim):
        super(NeuralNetworkLearner, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epochs = 100
        self.fitted = False
        self._setup_model()

    def _setup_model(self):

        self.model = Sequential()
        self.model.add(Dense(32, input_dim=self.input_dim, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.output_dim, activation='softmax'))
        self.epochs = self.epochs

        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, x, y):
        # One hot encoding
        yc = keras.utils.to_categorical(y, num_classes=self.output_dim)
        # Compile
        self._setup_model()
        # Train
        self.model.fit(x, yc, epochs=self.epochs, batch_size=64,
                       verbose=0)
        self.fitted = True

    def predict(self, x):

        if not self.fitted:
            raise NotFittedError
        p = self.model.predict(x)
        return np.argmax(p, axis=1)

    def predict_proba(self, x):
        return self.model.predict(x)


class RandomForestLearner(Learner):
    """docstring for LogisticRegressionLearner"""

    def __init__(self, input_dim, output_dim):
        super(RandomForestLearner, self).__init__()
        n_estimators = 50
        max_depth = 5
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class LowBiasRandomForestLearner(Learner):
    """docstring for LogisticRegressionLearner"""

    def __init__(self, input_dim, output_dim):
        super(LowBiasRandomForestLearner, self).__init__()
        self.model = RandomForestClassifier(n_estimators=100, max_depth=None)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class LogisticRegressionLearner(Learner):
    """docstring for LogisticRegressionLearner"""

    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionLearner, self).__init__()
        self.model = LogisticRegression()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)
