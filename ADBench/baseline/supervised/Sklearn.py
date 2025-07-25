from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

class SklearnSupervisedModel():
    def __init__(self, config, model_name):
        self.config = config
        self.model_name = model_name
        self.model_dict = {
            'nb': GaussianNB,
            'svm': SVC,
            'mlp': MLPClassifier,
            'rf': RandomForestClassifier,
            'lgb': lgb.LGBMClassifier,
            'xgb': xgb.XGBClassifier,
            'catb': CatBoostClassifier
        }

        if model_name.lower() not in self.model_dict:
            raise ValueError(f"Model {model_name} is not supported.")

        self.model_class = self.model_dict[model_name.lower()]
        self.model_params = self.config.get(model_name.lower(), {}).get("parameters", {})
        self.model = None

        if model_name.lower() == "svm" and "probability" not in self.model_params:
            self.model_params["probability"] = True

    def fit(self, X_train, y_train):
        print(f"Fitting {self.model_name} with parameters: {self.model_params}")
        self.model = self.model_class(**self.model_params)
        self.model.fit(X_train, y_train)
        return self

    def predict_score(self, X):
        if not self.model:
            raise RuntimeError("Model has not been fitted yet.")
        score = self.model.predict_proba(X)[:, 1]
        return score