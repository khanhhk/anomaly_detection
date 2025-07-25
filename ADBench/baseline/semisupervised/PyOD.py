from pyod.models.xgbod import XGBOD

class PYODSemisupervisedModel():
    def __init__(self, config, model_name):
        self.config = config
        self.model_name = model_name
        self.model_dict = {
            'xgbod': XGBOD,
        }
        if model_name.lower() not in self.model_dict:
            raise ValueError(f"Model {model_name} is not supported.")

        self.model_class = self.model_dict[model_name.lower()]
        self.model_params = self.config.get(model_name.lower(), {}).get("parameters", {})
        self.model = None

    def fit(self, X_train, y_train):
        print(f"Fitting XGBOD with parameters: {self.model_params}")
        self.model = XGBOD(**self.model_params)
        self.model.fit(X_train, y_train)
        return self

    def predict_score(self, X):
        if not self.model:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.decision_function(X)