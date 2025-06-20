import numpy as np
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from adbench.myutils import Utils

class DataGenerator():
    def __init__(self, dataset_path: str, seed: int = 42, test_size: float = 0.3):
        self.seed = seed
        self.test_size = test_size
        self.dataset_path = dataset_path
        self.utils = Utils()

    def generator(self, minmax=True,
                  la=None, at_least_one_labeled=False):
        self.utils.set_seed(self.seed)

        # Load dữ liệu 
        if not self.dataset_path.endswith(".npz"):
            raise ValueError("File dataset phải có định dạng .npz")

        try:
            data = np.load(self.dataset_path)
            X = data['X']
            y = data['y']
        except Exception as e:
            raise IOError(f"Không thể load file .npz: {e}")

        # Tính số anomaly được gán nhãn
        if isinstance(la, float):
            n_labeled_anomalies = ceil(sum(y) * (1 - self.test_size) * la) if at_least_one_labeled else int(
                sum(y) * (1 - self.test_size) * la)
        elif isinstance(la, int):
            n_labeled_anomalies = la
        else:
            raise ValueError("Tham số 'la' phải là float hoặc int")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            stratify=y, shuffle=True)

        if minmax:
            scaler = MinMaxScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        # Chọn index anomaly
        idx_normal = np.where(y_train == 0)[0]
        idx_anomaly = np.where(y_train == 1)[0]

        if isinstance(la, float):
            num_to_select = min(ceil(la * len(idx_anomaly)), len(idx_anomaly)) if at_least_one_labeled else int(
                la * len(idx_anomaly))
        else:
            if la > len(idx_anomaly):
                raise ValueError(f"Số lượng anomalies được gán nhãn ({la}) vượt quá số anomalies hiện có ({len(idx_anomaly)})")
            num_to_select = la

        idx_labeled_anomaly = np.random.choice(idx_anomaly, num_to_select, replace=False)
        idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
        idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)

        # Gán nhãn
        y_train[idx_unlabeled] = 0
        y_train[idx_labeled_anomaly] = 1

        self.utils.data_description(X_train, y_train)

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
