from adbench.myutils import Utils
from pyod.models.xgbod import XGBOD

class PYOD:
    def __init__(self, seed, tune=False):
        self.seed = seed
        self.utils = Utils()
        self.tune = tune
        self.model = None

    def fit(self, X_train, y_train, ratio=None):
        self.utils.set_seed(self.seed)
        self.model = XGBOD().fit(X_train, y_train)
        return self

    def predict_score(self, X):
        return self.model.decision_function(X)
    