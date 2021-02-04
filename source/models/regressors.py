"""
Module containing all the regressive models considered.
"""

import numpy as np
import cvxpy as cp

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import NotFittedError

from keras import backend as K
from keras.models import Input
from keras.layers import Dense
from keras.models import Model as NNModel

from source.macs import Learner

import warnings
warnings.filterwarnings("ignore")

# ======================================================================
# Regressor with fairness penalty term.
# ======================================================================


class FairRegressor():
    """
    """

    def __init__(self, alpha, I_train, gamma=0):

        super(FairRegressor, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.I = I_train
        self.w = None

    def mse(self, x, y, w):

        y = np.array(y).flatten()

        return cp.sum_squares(x @ w - y) / len(y)

    def fit(self, x, y):
        n_points = x.shape[0]
        n_feat = x.shape[1]
        w = cp.Variable(shape=n_feat)
        constraints = []

        fairness = 0
        for key, val in self.I.items():
            Np = np.sum(val)
            if Np > 0:
                tmp = (x @ w) / n_points - cp.multiply(val, (x @ w)) / Np
                fairness += cp.abs(cp.sum(tmp))

        mse = self.mse(x, y, w)
        fair_reg = self.alpha * fairness
        w_reg = self.gamma * cp.sum_squares(w)

        obj_fct = mse + fair_reg + w_reg

        prob = cp.Problem(cp.Minimize(obj_fct), constraints)
        # prob.solve()
        prob.solve(solver='ECOS')
        print("Solution status: " + str(prob.status))

        print("Value of MSE: %.3f" % mse.value)
        print("Value of fairness regularization: %.3f" % fairness.value)

        self.w = w.value

    def predict(self, x):

        if self.w is None:
            raise NotFittedError

        return x.dot(self.w)


# ======================================================================
# Learners
# ======================================================================

class LRegressor(Learner):
    """
    Basically the scikit-learn regressor with some wrapping.
    """
    def __init__(self):

        super(LRegressor, self).__init__()
        self.model = LinearRegression()

    def fit(self, x, y):

        self.model.fit(x, y)

    def predict(self, x):

        return self.model.predict(x)


class GBTree(Learner):
    """
    """

    def __init__(self):

        super(GBTree, self).__init__()
        n_estimators = 50
        min_samples_leaf = 5
        self.model = GradientBoostingRegressor(n_estimators=n_estimators,
                                               min_samples_leaf=min_samples_leaf)

    def fit(self, x, y):

        self.model.fit(x, y)

    def predict(self, x):

        yhat = self.model.predict(x)
        return yhat


class LowBiasRandomForestLearner(Learner):
    """docstring for LogisticRegressionLearner"""

    def __init__(self):
        super(LowBiasRandomForestLearner, self).__init__()
        self.model = RandomForestRegressor(n_estimators=100, max_depth=None)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        raise NotImplementedError

class Net(Learner):
    """
    """

    def __init__(self, input_dim, output_dim):

        super(Net, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_layer_neurons = 32
        self.outermost_layer_neurons = 32
        self.optimizer = 'rmsprop'
        self.epochs = 100
        self.batch_size = 64
        self.fitted = False
        self._setup_model()

    def _setup_model(self):

        # Model initialization.
        model_input = Input(shape=self.input_dim, name='input')
        layer1 = Dense(self.inner_layer_neurons, activation='relu', name='dense1')(model_input)
        layer2 = Dense(self.inner_layer_neurons, activation='relu', name='dense2')(layer1)
        model_output = Dense(self.output_dim, activation='linear', name='output', use_bias=True)(layer2)
        self.model = NNModel(inputs=model_input, outputs=model_output)
        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

    def get_inner_features(self, x):
        """
        Given a model and a set of inputs, compute the corresponding features on the outermost layer.
        """
        inp = self.model.layers[0].input  # input layer
        output = self.model.layers[-2].output  # outermost layer outputs
        functor = K.function([inp, K.learning_phase()], [output])  # evaluation function

        in_features = list()
        for _x in x:
            layer_out = functor([_x[np.newaxis, :], 0])[0]
            phi = layer_out.T.ravel()
            in_features.append(phi)

        return in_features

    def fit(self, x, y):

        self._setup_model()
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
        self.fitted = True

    def predict(self, x):
        """
        Wrapper for the network predict function, to return classes instead of probabilities.
        """
        if not self.fitted:
            raise NotFittedError
        return self.model.predict(x)

    def postprocessing(self, x, y, D, I):

        raise NotImplementedError()
        train_preds = self.predict(x)
        ann_inner_features = self.get_inner_features(x)
        qp = QuadProb(self.params).compute_opt_weights(y, train_preds, D,
                                                       I, 0, ann_inner_features)
        opt_weights = np.array(list(qp.values())).reshape(-1, 1)

        ann_weights = self.model.get_weights()
        ll_shape = ann_weights[-1].shape
        opt_ann_weights = ann_weights.copy()
        opt_ann_weights[-1] = np.reshape(opt_weights, ll_shape)
        self.model.set_weights(opt_ann_weights)
