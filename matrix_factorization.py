from interface import Regressor
from utils import Config, get_data
from linear_regression_baseline import Baseline
import numpy as np
from config import USER_COL_NAME_IN_DATAEST, ITEM_COL_NAME_IN_DATASET, RATING_COL_NAME_IN_DATASET

class MatrixFactorization(Regressor):
    def __init__(self, config):
        self.k = config.k
        self.number_bias_epochs = int(0.25*config.epochs)
        self.global_mean = None
        self.user_biases = None
        self.item_biases = None
        self.U = None
        self.V = None
        self.gamma = config.gamma
        self.lr = config.lr
        self.epochs = config.epochs

    def record(self, epoch, covn_dict):
        print(f"epoch # {epoch} : {covn_dict}")

    def calc_regularization(self):
        return self.gamma*(np.sum(self.user_biases**2) + np.sum(self.item_biases**2) + np.sum(self.U**2) + np.sum(self.V**2))

    def fit(self, X):
        self.global_mean = X[RATING_COL_NAME_IN_DATASET].mean()
        self.U = np.random.normal(scale=0.2 / self.k, size=(X[USER_COL_NAME_IN_DATAEST].max()+1, self.k))
        self.V = np.random.normal(scale=0.2 / self.k, size=(X[ITEM_COL_NAME_IN_DATASET].max()+1, self.k))
        self.user_biases = np.random.normal(scale=0.2 / self.k, size=(X[USER_COL_NAME_IN_DATAEST].max()+1))
        self.item_biases = np.random.normal(scale=0.2 / self.k, size=(X[ITEM_COL_NAME_IN_DATASET].max()+1))
        train = X.values
        for epoch in range(1, self.epochs + 1):
            print(f'starting epoch {epoch}')
            self.run_epoch(train, epoch)
            train_accuracy = self.calculate_rmse(train)
            train_objective = train_accuracy + self.calc_regularization()
            convergence_params = {'train_accuracy': train_accuracy, 'train_objective': train_objective}
            self.record(epoch, convergence_params)

    def run_epoch(self, data: np.array, epoch):
        # if epoch > 1:
        #     self.lr = self.lr * 0.95
        np.random.shuffle(data)
        for row in data:
            user, item, rating = row
            prediction = self.predict_on_pair(user, item)
            error = rating - prediction
            self.user_biases[user] += self.lr * (error - self.gamma * self.user_biases[user])
            self.item_biases[item] += self.lr * (error - self.gamma * self.item_biases[item])
            if epoch > self.number_bias_epochs:
                self.U[user, :] += self.lr * (error * self.V[item, :] - self.gamma * self.U[user, :])
                self.V[item, :] += self.lr * (error * self.U[user, :] - self.gamma * self.V[item, :])

    def predict_on_pair(self, user, item):
        try:
            user_item_bias = self.global_mean + self.user_biases[user] + self.item_biases[item]
            prediction = np.clip(user_item_bias + self.U[user, :].dot(self.V[item, :].T), 1, 5)
        except:
            if user in self.data[USER_COL_NAME_IN_DATAEST]:
                prediction = self.data.loc[self.data[USER_COL_NAME_IN_DATAEST] == user][RATING_COL_NAME_IN_DATASET].mean()
            else:
                prediction = self.global_mean
        return prediction


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

# decay of 0.9 --> 0.9543123124982653
# decay of 0.95 --> 0.9463740389559875
# decay of 1.0 --> 0.9298985871120552
