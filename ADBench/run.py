import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import time
import gc
import os
from keras import backend as K
from sklearn.model_selection import train_test_split
from math import ceil
from adbench.myutils import Utils

class RunPipeline():
    def __init__(self, suffix:str=None, mode:str='rla', parallel:str=None):
        '''
        :param suffix: saved file suffix (including the model performance result and model weights)
        :param mode: rla or nla —— ratio of labeled anomalies or number of labeled anomalies
        :param parallel: unsupervise, semi-supervise or supervise, choosing to parallelly run the code
        :param noise_type: duplicated_anomalies, irrelevant_features or label_contamination —— whether to test the model robustness
        '''

        # utils function
        self.utils = Utils()
        self.mode = mode
        self.parallel = parallel

        # the suffix of all saved files
        self.suffix = suffix or 'results'

        # ratio of labeled anomalies
        self.rla_list = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
        # number of labeled anomalies
        self.nla_list = [0, 1, 5, 10, 25, 50, 75, 100]
        self.seed_list = [1, 2, 3]
        # model_dict (model_name: clf)
        self.model_dict = {}

        # unsupervised algorithms
        if self.parallel == 'unsupervised':
            from baseline.unsupervised.PyOD import PyODUnsupervisedModel
            from baseline.unsupervised.DAGMM.run import DAGMM

            # from pyod
            for _ in ['IForest', 'OCSVM', 'CBLOF', 'COF', 'COPOD', 'ECOD', 'HBOS', 'KNN', 'LODA',
                      'LOF', 'PCA', 'SOD', 'DeepSVDD']:
                self.model_dict[_] = PyODUnsupervisedModel

            # DAGMM
            self.model_dict['DAGMM'] = DAGMM

        # semi-supervised algorithms
        elif self.parallel == 'semisupervised':
            from baseline.semisupervised.PyOD import PYODSemisupervisedModel
            from baseline.semisupervised.GANomaly.run import GANomaly
            from baseline.semisupervised.DeepSAD.src.run import DeepSAD
            from baseline.semisupervised.REPEN.run import REPEN
            from baseline.semisupervised.DevNet.run import DevNet
            from baseline.semisupervised.PReNet.run import PReNet
            from baseline.semisupervised.FEAWAD.run import FEAWAD

            self.model_dict = {'GANomaly': GANomaly,
                               'DeepSAD': DeepSAD,
                               'REPEN': REPEN,
                               'DevNet': DevNet,
                               'PReNet': PReNet,
                               'FEAWAD': FEAWAD,
                               'XGBOD': PYODSemisupervisedModel}

        # fully-supervised algorithms
        elif self.parallel == 'supervised':
            from baseline.supervised.Sklearn import SklearnSupervisedModel
            from baseline.supervised.FTTransformer.run import FTTransformer

            # from sklearn
            for _ in ['NB', 'SVM', 'MLP', 'RF', 'LGB', 'XGB', 'CatB']:
                self.model_dict[_] = SklearnSupervisedModel
            # ResNet and FTTransformer for tabular data
            for _ in ['ResNet', 'FTTransformer']:
                self.model_dict[_] = FTTransformer
        else:
            raise NotImplementedError

    def process_data(self, X, y, la, seed):
        np.random.seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

        idx_normal = np.where(y_train == 0)[0]
        idx_anomaly = np.where(y_train == 1)[0]

        if isinstance(la, float):
            idx_labeled_anomaly = np.random.choice(idx_anomaly, max(1, int(la * len(idx_anomaly))), replace=False)
        elif isinstance(la, int):
            if la > len(idx_anomaly):
                raise ValueError("la > number of anomalies")
            idx_labeled_anomaly = np.random.choice(idx_anomaly, la, replace=False)
        else:
            raise NotImplementedError

        idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
        idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)

        y_train[idx_unlabeled] = 0
        y_train[idx_labeled_anomaly] = 1

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    
    # model fitting function
    def model_fit(self):
        try:
            if self.model_name in ['DevNet', 'FEAWAD', 'REPEN']:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name, save_suffix=self.suffix)
            else:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name)
        except Exception as error:
            print(f'Error in model initialization. Model:{self.model_name}, Error: {error}')
            return None, None, {'aucroc': np.nan, 'aucpr': np.nan}

        try:
            start_time = time.time()
            self.clf = self.clf.fit(X_train=self.data['X_train'], y_train=self.data['y_train'])
            end_time = time.time(); time_fit = end_time - start_time

            start_time = time.time()
            if self.model_name == 'DAGMM':
                score_test = self.clf.predict_score(self.data['X_train'], self.data['X_test'])
            else:
                score_test = self.clf.predict_score(self.data['X_test'])
            end_time = time.time(); time_inference = end_time - start_time

            result = self.utils.metric(y_true=self.data['y_test'], y_score=score_test)

            K.clear_session()
            print(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")

            del self.clf
            gc.collect()

        except Exception as error:
            print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
            return None, None, {'aucroc': np.nan, 'aucpr': np.nan}

        return time_fit, time_inference, result

    # run the experiments in ADBench
    def run(self, dataset=None, clf=None):
        assert dataset is not None and 'X' in dataset and 'y' in dataset
        X, y = dataset['X'], dataset['y']
        dataset_list = [None]

        if self.mode == 'nla':
            param_list = list(product(dataset_list, self.nla_list, self.seed_list))
        else:
            param_list = list(product(dataset_list, self.rla_list, self.seed_list))

        experiment_params = pd.MultiIndex.from_tuples(
            param_list,
            names=["dataset", "la", "seed"]
        )

        print(f'{len(self.model_dict.keys())} models will be evaluated.')
        print(f"Experiment results are saved at: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')}")
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result'), exist_ok=True)
        columns = list(self.model_dict.keys()) if clf is None else ['Customized']
        df_AUCROC = pd.DataFrame(index=experiment_params, columns=columns)
        df_AUCPR = pd.DataFrame(index=experiment_params, columns=columns)
        df_time_fit = pd.DataFrame(index=experiment_params, columns=columns)
        df_time_inference = pd.DataFrame(index=experiment_params, columns=columns)

        results = []
        for i, params in tqdm(enumerate(experiment_params)):
            _, la, self.seed = params

            if self.parallel == 'unsupervised' and self.mode == 'rla' and la != 0.0:
                continue

            self.data = self.process_data(X, y, la, seed=self.seed)

            if clf is None:
                for model_name in tqdm(self.model_dict.keys()):
                    self.model_name = model_name
                    self.clf = self.model_dict[self.model_name]
                    time_fit, time_inference, metrics = self.model_fit()
                    results.append([params, model_name, metrics, time_fit, time_inference])

                    df_AUCROC.loc[params, model_name] = metrics['aucroc']
                    df_AUCPR.loc[params, model_name] = metrics['aucpr']
                    df_time_fit.loc[params, model_name] = time_fit
                    df_time_inference.loc[params, model_name] = time_inference

            else:
                self.clf = clf; self.model_name = 'Customized'
                time_fit, time_inference, metrics = self.model_fit()
                results.append([params, self.model_name, metrics, time_fit, time_inference])
                df_AUCROC.loc[params, self.model_name] = metrics['aucroc']
                df_AUCPR.loc[params, self.model_name] = metrics['aucpr']
                df_time_fit.loc[params, self.model_name] = time_fit
                df_time_inference.loc[params, self.model_name] = time_inference

        df_AUCROC.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'AUCROC_' + self.suffix + '.csv'))
        df_AUCPR.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'AUCPR_' + self.suffix + '.csv'))
        df_time_fit.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'Time(fit)_' + self.suffix + '.csv'))
        df_time_inference.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'Time(inference)_' + self.suffix + '.csv'))

        return results