from typing import Dict
import pickle

import numpy as np
import pandas as pd

from interface import Regressor
from utils import Config, get_data
from config import USER_COL_NAME_IN_DATAEST, ITEM_COL_NAME_IN_DATASET, RATING_COL_NAME_IN_DATASET, BASELINE_PARAMS_FILE_PATH


class Baseline(Regressor):
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.train_epochs = config.epochs
        self.epoch_size = config.epoch_size
        self.n_users = None
        self.n_items = None
        self.user_map = None
        self.item_map = None
        self.user_biases = None  # b_u (users) vector
        self.item_biases = None  # # b_i (items) vector
        self.current_epoch = 0
        self.global_bias = None

    def record(self, covn_dict: Dict):
        epoch = "{:02d}".format(self.current_epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.4}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

    def calc_regularization(self):
        user_norm = np.linalg.norm(self.user_biases)
        item_norm = np.linalg.norm(self.item_biases)
        return self.lr * (user_norm + item_norm)

    def fit(self, X):
            self.n_users = X[USER_COL_NAME_IN_DATAEST].unique().size
            self.n_items = X[ITEM_COL_NAME_IN_DATASET].unique().size
            self.user_biases = np.zeros(self.n_users)
            self.item_biases = np.zeros(self.n_items)
            self.user_map = pd.Series(X[USER_COL_NAME_IN_DATAEST].unique())
            self.item_map = pd.Series(X[ITEM_COL_NAME_IN_DATASET].unique())
            self.global_bias = X[RATING_COL_NAME_IN_DATASET].mean()
            while self.current_epoch < self.train_epochs:
                self.run_epoch(X.to_numpy())
                train_mse = np.square(self.calculate_rmse(X.to_numpy()))
                train_objective = train_mse * X.shape[0] + self.calc_regularization()
                epoch_convergence = {"train_objective": train_objective,
                                     "train_mse": train_mse}
                self.record(epoch_convergence)
                self.current_epoch += 1
            self.save_params()


    def run_epoch(self, data: np.array):
        sub_data = data[np.random.choice(data.shape[0], self.epoch_size)]
        for row in sub_data:
            user, item, rating = row
            user_index, item_index = self.user_map[self.user_map == user].index[0], self.item_map[self.item_map == item].index[0]
            prediction = self.global_bias + self.user_biases[user_index] + self.item_biases[item_index]
            error = rating - prediction
            self.user_biases[user_index] += self.lr * (error - self.gamma * self.user_biases[user_index])
            self.item_biases[item_index] += self.lr * (error - self.gamma * self.item_biases[item_index])


    def predict_on_pair(self, user: int, item: int):
        try:
            prediction = self.global_bias + self.user_biases[self.user_map[self.user_map == user].index[0]] + self.item_biases[self.item_map[self.item_map == item].index[0]]
        except:
            prediction = self.global_bias
        return prediction


    def calculate_rmse(self, data: np.array):
        #rating_pred = np.array([self.predict_on_pair(u_ind, i_ind) for u_ind, i_ind in zip(self.user_map, self.item_map)])
        e = 0
        for row in data:
            user, item, rating = row
            e += np.square(rating - self.predict_on_pair(user, item))
        return np.sqrt(e / data.shape[0])


    def save_params(self):
        pickle.dump(baseline_model, open(BASELINE_PARAMS_FILE_PATH, 'wb'))
        #loaded_model = pickle.load(open(BASELINE_PARAMS_FILE_PATH, 'rb')) load


if __name__ == '__main__':
    baseline_config = Config(
        lr=0.001,
        gamma=0.001,
        epochs=10,
        epoch_size=500)
    train, validation = get_data()
    baseline_model = Baseline(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
