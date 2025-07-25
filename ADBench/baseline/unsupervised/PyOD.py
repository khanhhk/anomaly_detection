from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.deep_svdd import DeepSVDD


class PyODUnsupervisedModel():
    def __init__(self, config, model_name):
        '''
        :param config: config from yaml file, contains parameters for the model
        :param model_name: model name
        '''
        self.config = config
        self.model_name = model_name
        self.model_dict = {
            'iforest': IForest,
            'ocsvm': OCSVM,
            'cblof': CBLOF,
            'cof': COF,
            'copod': COPOD,
            'ecod': ECOD,
            'hbos': HBOS,
            'knn': KNN,
            'loda': LODA,
            'lof': LOF,
            'pca': PCA,
            'sod': SOD,
            'deepsvdd': DeepSVDD
        }
        
        if model_name.lower() not in self.model_dict:
            raise ValueError(f"Model {model_name} is not supported.")

        self.model_class = self.model_dict[model_name.lower()]
        self.model_params = self.config.get(model_name.lower(), {}).get("parameters", {})
        self.model = None

    def fit(self, X):
        print(f"Fitting {self.model_name} with parameters: {self.model_params}")
        self.model = self.model_class(**self.model_params)
        self.model.fit(X)
        return self

    # from pyod: for consistency, outliers are assigned with larger anomaly scores
    def predict_score(self, X):
        if not self.model:
            raise RuntimeError("Model has not been fitted yet.")
        score = self.model.decision_function(X)
        return score