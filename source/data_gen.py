"""
Module containing the loader of all dataset.
"""

import os
import pandas as pd
import numpy as np
from source import utils

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ShuffleSplit

_resource_folder = 'resources'


BALANCE_DATASET = [
    'iris',
    'whitewine',
    'redwine',
    'shuttle',
    'dota'
]

# ======================================================================
# Classification data.
# ======================================================================


def load_iris_data():
    # Load raw data
    fname = os.path.join(_resource_folder, 'iris.csv')
    data = pd.read_csv(fname, sep=',')
    # # Scalability test
    # data = pd.DataFrame(data=np.repeat(data.values, 100, axis=0), columns=data.columns)
    # Shuffle
    data = data.sample(frac=1, random_state=42)
    # Separate input
    x = data[[c for c in data.columns if c != 'class']]
    # Separate output  (as integers)
    y = data['class'].astype('category').cat.codes
    # Return
    return x.values, y.values


def load_winequality_red_data():
    # Load raw data
    fname = os.path.join(_resource_folder, 'winequality-red.csv')
    data = pd.read_csv(fname, sep=';')
    # # Scalability test
    # data = pd.DataFrame(data=np.repeat(data.values, 100, axis=0), columns=data.columns)
    # Shuffle
    data = data.sample(frac=1, random_state=42)
    # Separate input
    x = data[[c for c in data.columns if c != 'quality']]
    # Separate output  (as integers)
    y = data['quality'].astype('category').cat.codes
    # Return
    return x.values, y.values


def load_winequality_white_data():
    # Load raw data
    fname = os.path.join(_resource_folder, 'winequality-white.csv')
    data = pd.read_csv(fname, sep=';')
    # # Scalability test
    # data = pd.DataFrame(data=np.repeat(data.values, 100, axis=0), columns=data.columns)
    # Shuffle
    data = data.sample(frac=1, random_state=42)
    # Separate input
    x = data[[c for c in data.columns if c != 'quality']]
    # Separate output  (as integers)
    y = data['quality'].astype('category').cat.codes
    # Return
    return x.values, y.values


def load_shuttle_data():
    # Load raw data
    fname = os.path.join(_resource_folder, 'shuttle.trn')
    data = pd.read_csv(fname, sep=' ', header=None,
            names=['time', '1', '2', '3', '4', '5', '6', '7', '8', 'class'])
    # Shuffle
    data = data.sample(frac=1, random_state=42)
    # Separate input
    x = data[[c for c in data.columns if c != 'class']]
    # Separate output  (as integers)
    y = data['class'].astype('category').cat.codes
    # Return
    return x.values, y.values


def load_dota_data():
    # Load raw data
    fname = os.path.join(_resource_folder, 'dota2Train.csv')
    data = pd.read_csv(fname, sep=',', header=None)
    # Shuffle
    data = data.sample(frac=1, random_state=42)
    # Separate input
    # NOTE the "cluster id" column is discarded for simplicity
    x = data[[c for c in data.columns if c not in (0, 1, 2, 3)]]
    # Separate output  (as integers)
    y = data[0].astype('category').cat.codes
    # Return
    return x.values, y.values


# ======================================================================
# Fair Regression data.
# ======================================================================


def load_crime():
    """
    Read data from csv file and return them as a numpy arrays.
    """

    # LOAD DATA FROM FILE.
    # filename = "resources\CommViolPredUnnormalizedData.csv"
    filename = os.path.join('resources', 'CommViolPredUnnormalizedData.csv')
    data = pd.read_csv(filename, header=0, sep=';', na_values='?', skipinitialspace=True)
    data = data.sample(frac=1, random_state=42)

    targets = ['violentPerPop']
    pfeatures = ['race']

    # Drop rows with no associated attribute to be predicted.
    dataset = data.dropna(subset=targets, axis=0).reset_index(drop=True)

    # Keep only features that have more than 95% of points with associated value.
    features_to_drop = list()
    n_points = len(dataset)
    acc_rate = 0.95

    for c in dataset.columns:
        tot_values = np.sum(dataset[c].isna())
        if tot_values >= (1 - acc_rate) * n_points:
            features_to_drop.append(c)

    dataset = dataset.drop(features_to_drop, axis=1)

    # Remove features that are either correlated with the target or useless.
    feat_to_remove = [
        'fold',
        'communityname',
        'state',
        'murders',
        'murdPerPop',
        'rapes',
        'rapesPerPop',
        'robberies',
        'robbbPerPop',
        'assaults',
        'assaultPerPop',
        'burglaries',
        'burglPerPop',
        'larcenies',
        'larcPerPop',
        'autoTheft',
        'autoTheftPerPop',
        'arsons',
        'arsonsPerPop',
        'nonViolPerPop'
    ]

    feat_to_remove += targets + pfeatures

    # Prepare the feature dataset.
    features = [f for f in dataset.columns if f not in feat_to_remove]
    dataset = dataset[features + pfeatures + targets]

    # Last check on Nan values.
    dataset = dataset.dropna(axis=0).reset_index(drop=True)

    # Force all types to float.
    for c in dataset.columns:
        dataset[c] = dataset[c].astype(float)

    # Features selection.
    top_features = utils.get_top_features(dataset[features], dataset[targets], n=15)

    for pfeat in pfeatures:
        if pfeat in top_features:
            print("Protected feature " + pfeat + " in top features!")

    x, xp, y = dataset[top_features].values, dataset[pfeatures].values, dataset[targets].values

    return x, xp, y

# ======================================================================
# Fair Classification data.
# ======================================================================


def load_adult():
    """
    Read data from csv file and return them as a pandas DataFrame.
    """

    # LOAD DATA FROM FILE.
    filename = "resources//adult.csv"
    data = pd.read_csv(filename, header=0, sep=';', na_values='?', skipinitialspace=True)

    # Drop rows with no associated attribute to be predicted.
    dataset = data.dropna(axis=0).reset_index(drop=True)

    # Target.
    targets = ['income']
    # Protected features.
    pfeatures = ['race']

    lb = LabelBinarizer()
    _df = lb.fit_transform(dataset[targets].values)
    target_df = pd.DataFrame(_df, columns=targets)

    # Features to drop.
    feat_to_remove = [
        'education',
        'native-country',
    ]
    feat_to_remove += targets + pfeatures

    # Prepare the feature dataset.
    features = [f for f in dataset.columns if f not in feat_to_remove]
    feat_dataset = dataset[features]

    # One-hot encoder to be used.
    enc = OneHotEncoder(drop=None, dtype=np.float, sparse=False)

    # Encode the categorical features of the data.
    _df = feat_dataset[features].select_dtypes(include=['object']).copy()
    _label = list(_df.columns)
    onehot_x = enc.fit_transform(_df.values)
    oh_cat_features = enc.get_feature_names(_label).tolist()
    cat_df = pd.DataFrame(onehot_x, columns=oh_cat_features)

    # Select numerical features of the data.
    num_df = feat_dataset[features].select_dtypes(exclude=['object']).copy()
    num_features = list(num_df.columns)

    # Encode the protected features of the data.
    onehot_x = enc.fit_transform(dataset[pfeatures])
    oh_pfeatures = enc.get_feature_names(pfeatures).tolist()
    pfeat_df = pd.DataFrame(onehot_x, columns=oh_pfeatures)

    xnp = pd.concat([num_df, cat_df], axis=1).values
    xp = pfeat_df.values
    y = target_df.values

    return xnp, xp, y


dataset_loaders = {
    'iris': load_iris_data,
    'whitewine': load_winequality_white_data,
    'redwine': load_winequality_red_data,
    'shuttle': load_shuttle_data,
    'dota': load_dota_data,
    'adult': load_adult,
    'crime': load_crime,
}


class Dataset():
    """
    Class that loads and prepares a dataset.
    """
    def __init__(self, name, test_size):
        self._name = name
        self._test_size = test_size

        # Data for training
        self.xnp_tr = None
        self.xp_tr = None
        self.xnp_ts = None
        self.xp_ts = None
        self.y_tr = None
        self.y_ts = None

        if self._name not in dataset_loaders.keys():
            raise ValueError(f'Unknown dataset "{self._name}"')

        self._load_data()

    def _load_data(self):
        """ Load the data from file. """
        if self._name in BALANCE_DATASET:
            _loader = dataset_loaders[self._name]
            xnp, y = _loader()

            # Train - Test split
            gen = ShuffleSplit(n_splits=1, random_state=42, test_size=self._test_size).split(xnp)
            train_idx, test_idx = next(gen)

            # Train data.
            self.xnp_tr = xnp[train_idx]
            self.y_tr = y[train_idx]
            # Test data.
            self.xnp_ts = xnp[test_idx]
            self.y_ts = y[test_idx]

        else:
            _loader = dataset_loaders[self._name]
            xnp, xp, y = _loader()
            # self.xnp, self.xp, self.y = _loader()

            # Train - Test split
            gen = ShuffleSplit(n_splits=1, random_state=42, test_size=self._test_size).split(xnp)
            train_idx, test_idx = next(gen)

            # Train data.
            self.xnp_tr = xnp[train_idx]
            self.xp_tr = xp[train_idx]
            self.y_tr = y[train_idx]
            # Test data.
            self.xnp_ts = xnp[test_idx]
            self.xp_ts = xp[test_idx]
            self.y_ts = y[test_idx]


if __name__ == '__main__':
    dataset = Dataset('redwine', test_size=.2)
    print(dataset.xnp_tr, dataset.xp_tr, dataset.y_tr)
