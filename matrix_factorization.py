from interface import Regressor
from utils import Config, get_data


class MatrixFactorization(Regressor):
    def __init__(self, config):
        raise NotImplementedError

    def record(self, covn_dict):
        raise NotImplementedError

    def calc_regularization(self):
        raise NotImplementedError

    def fit(self, X):
        raise NotImplementedError

    def run_epoch(self):
        raise NotImplementedError

    def predict_on_pair(self, use, item):
        raise NotImplementedError


if __name__ == '__main__':
    baseline_config = Config(
        lr=0.01,
        gamma=0.001,
        k=24,
        epochs=10)

    train, validation = get_data()
    baseline_model = MatrixFactorization(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
