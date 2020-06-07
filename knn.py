import numpy as np

from interface import Regressor
from utils import get_data, Config


class KnnItemSimilarity(Regressor):
    def __init__(self, config):
        raise NotImplementedError

    def fit(self, X: np.array):
        raise NotImplementedError

    def build_item_to_itm_corr_dict(self, data):
        raise NotImplementedError

    def predict_on_pair(self, user, item):
        raise NotImplementedError

    def upload_params(self):
        raise NotImplementedError

    def save_params(self):
        raise NotImplementedError


if __name__ == '__main__':
    knn_config = Config(k=25)
    train, validation = get_data()
    knn = KnnItemSimilarity(knn_config)
    knn.fit(train)
    print(knn.calculate_rmse(validation))
