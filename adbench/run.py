import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import gc
import os
from keras import backend as K

from adbench.datasets.data_generator import DataGenerator
from adbench.myutils import Utils

class RunPipeline():
    def __init__(self, dataset_path: str, suffix: str = None, mode: str = 'rla', parallel: str = None, seed: int = 42):
        self.dataset_path = dataset_path
        self.mode = mode
        self.parallel = parallel
        self.suffix = suffix + '_' + self.parallel
        self.seed = seed
        self.utils = Utils()
        self.data_generator = DataGenerator(dataset_path=self.dataset_path, seed=self.seed)

        self.rla_list = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
        self.nla_list = [0, 1, 5, 10, 25, 50, 75, 100]

        self.model_dict = self._load_models()

    def _load_models(self):
        model_dict = {}
        if self.parallel == 'unsupervise':
            from adbench.baseline.unsupervised.PyOD import PYOD
            from adbench.baseline.unsupervised.DAGMM.run import DAGMM
            for name in ['IForest', 'OCSVM', 'CBLOF', 'COF', 'COPOD', 'ECOD', 'HBOS',
                         'KNN', 'LODA', 'LOF', 'PCA', 'SOD', 'DeepSVDD']:
                model_dict[name] = PYOD
            model_dict['DAGMM'] = DAGMM

        elif self.parallel == 'semi-supervise':
            from adbench.baseline.semisupervised.PyOD import PYOD
            from adbench.baseline.semisupervised.GANomaly.run import GANomaly
            from adbench.baseline.semisupervised.DeepSAD.src.run import DeepSAD
            from adbench.baseline.semisupervised.REPEN.run import REPEN
            from adbench.baseline.semisupervised.DevNet.run import DevNet
            from adbench.baseline.semisupervised.PReNet.run import PReNet
            from adbench.baseline.semisupervised.FEAWAD.run import FEAWAD
            model_dict = {
                'GANomaly': GANomaly, 'DeepSAD': DeepSAD, 'REPEN': REPEN,
                'DevNet': DevNet, 'PReNet': PReNet, 'FEAWAD': FEAWAD, 'XGBOD': PYOD
            }

        elif self.parallel == 'supervise':
            from adbench.baseline.supervised.Supervised import supervised
            from adbench.baseline.supervised.FTTransformer.run import FTTransformer
            for name in ['NB', 'SVM', 'MLP', 'RF', 'LGB', 'XGB', 'CatB']:
                model_dict[name] = supervised
            for name in ['ResNet', 'FTTransformer']:
                model_dict[name] = FTTransformer

        else:
            raise NotImplementedError(f"Unknown mode: {self.parallel}")
        return model_dict

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
            # fitting
            start_time = time.time()
            self.clf = self.clf.fit(X_train=self.data['X_train'], y_train=self.data['y_train'])
            time_fit = time.time() - start_time

            # predicting score (inference)
            start_time = time.time()
            if self.model_name == 'DAGMM':
                score_test = self.clf.predict_score(self.data['X_train'], self.data['X_test'])
            else:
                score_test = self.clf.predict_score(self.data['X_test'])
            time_inference = time.time() - start_time

            # performance
            result = self.utils.metric(y_true=self.data['y_test'], y_score=score_test)
            print(f"[{self.model_name}] AUC-ROC: {result['aucroc']:.4f}, AUC-PR: {result['aucpr']:.4f}")
            K.clear_session()

            del self.clf
            gc.collect()

        except Exception as e:
            print(f"[{self.model_name}] Error: {e}")
            return None, None, {'aucroc': np.nan, 'aucpr': np.nan}

        return time_fit, time_inference, result

    def run(self):
        param_list = self.nla_list if self.mode == 'nla' else self.rla_list

        print(f"Total models: {len(self.model_dict)} | Total runs: {len(param_list)}")
        result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
        os.makedirs(result_path, exist_ok=True)
        print(f"Experiment results are saved at: {result_path}")

        columns = list(self.model_dict.keys())
        df_AUCROC = pd.DataFrame(index=param_list, columns=columns)
        df_AUCPR = pd.DataFrame(index=param_list, columns=columns)
        df_time_fit = pd.DataFrame(index=param_list, columns=columns)
        df_time_infer = pd.DataFrame(index=param_list, columns=columns)

        results = []
        for la in tqdm(param_list):
            try:
                self.data = self.data_generator.generator(la=la, at_least_one_labeled=True)
            except Exception as e:
                print(f"[DataGenerator] Error: {e}")
                continue

            for model_name in tqdm(self.model_dict.keys()):
                self.model_name = model_name
                self.clf = self.model_dict[model_name]

                time_fit, time_inference, metrics = self.model_fit()
                print(f"Done {model_name} | la={la} | AUCROC={metrics['aucroc']}, AUCPR={metrics['aucpr']}")

                results.append([la, model_name, metrics, time_fit, time_inference])

                df_AUCROC.at[la, model_name] = metrics['aucroc']
                df_AUCPR.at[la, model_name] = metrics['aucpr']
                df_time_fit.at[la, model_name] = time_fit
                df_time_infer.at[la, model_name] = time_inference

        print(df_AUCROC.head())
        df_AUCROC.to_csv(os.path.join(result_path, f"AUCROC_{self.suffix}.csv"))
        df_AUCPR.to_csv(os.path.join(result_path, f"AUCPR_{self.suffix}.csv"))
        df_time_fit.to_csv(os.path.join(result_path, f"Time(fit)_{self.suffix}.csv"))
        df_time_infer.to_csv(os.path.join(result_path, f"Time(inference)_{self.suffix}.csv"))

        return results
