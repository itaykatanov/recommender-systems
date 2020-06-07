import numpy as np

from interface import Regressor
from utils import get_data, Config


class KnnBaseline(Regressor):
    def __init__(self, config):
        raise NotImplementedError

    def fit(self, X: np.array):
        raise NotImplementedError

    def predict_on_pair(self, user: int, item: int):
        raise NotImplementedError

    def upload_params(self):
        raise NotImplementedError

    def save_params(self):
        raise NotImplementedError


if __name__ == '__main__':
    baseline_knn_config = Config(k=25)
    train, validation = get_data()
    knn_baseline = KnnBaseline(baseline_knn_config)
    knn_baseline.fit(train)
    print(knn_baseline.calculate_rmse(validation))
