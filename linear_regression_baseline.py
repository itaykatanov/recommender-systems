from typing import Dict
import pickle
import numpy as np
from interface import Regressor
from utils import Config, get_data
from config import USER_COL_NAME_IN_DATAEST, RATING_COL_NAME_IN_DATASET, BASELINE_PARAMS_FILE_PATH

class Baseline(Regressor):
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.train_epochs = config.epochs
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        self.current_epoch = 0

    def record(self, covn_dict: Dict):
        epoch = "{:02d}".format(self.current_epoch+1)
        temp = f"| epoch   # {epoch} :"
        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.4}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

    def calc_regularization(self):
        return self.gamma * (np.sum(self.user_biases**2) + np.sum(self.item_biases**2))

    def fit(self, X):
        self.user_biases = np.zeros(X[USER_COL_NAME_IN_DATAEST].max()+1)
        self.item_biases = np.zeros(X[USER_COL_NAME_IN_DATAEST].max()+1)
        self.global_mean = X[RATING_COL_NAME_IN_DATASET].mean()
        while self.current_epoch < self.train_epochs:
            print("Running Epoch #" + str(self.current_epoch + 1))
            self.run_epoch(X.to_numpy())
            train_mse = np.square(self.calculate_rmse(X.to_numpy()))
            train_objective = train_mse * X.shape[0] + self.calc_regularization()
            epoch_convergence = {"train_objective": train_objective, "train_mse": train_mse}
            self.record(epoch_convergence)
            self.current_epoch += 1
        self.save_params()

    def run_epoch(self, data: np.array):
        if (self.current_epoch > 1):
            self.lr = self.lr * 0.9
        for row in data:
            user, item, rating = row
            prediction = self.global_mean + self.user_biases[user] + self.item_biases[item]
            error = rating - prediction
            self.user_biases[user] += self.lr * (error - self.gamma * self.user_biases[user])
            self.item_biases[item] += self.lr * (error - self.gamma * self.item_biases[item])

    def predict_on_pair(self, user: int, item: int):
        try:
            prediction = self.global_mean + self.user_biases[user] + self.item_biases[item]
        except:
            prediction = self.global_mean
        return np.clip(prediction, 1, 5)

    def save_params(self):
        pickle.dump(baseline_model, open(BASELINE_PARAMS_FILE_PATH, 'wb'))


if __name__ == '__main__':
    baseline_config = Config(
        lr=0.001,
        gamma=0.001,
        epochs=10)
    train, validation = get_data()
    baseline_model = Baseline(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
