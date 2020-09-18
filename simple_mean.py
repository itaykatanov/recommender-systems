from interface import Regressor
from utils import get_data

class SimpleMean(Regressor):
    def __init__(self):
        self.user_means = {}

    def fit(self, X):
        self.user_means = X.groupby(['User_ID_Alias']).mean()['Ratings_Rating'].to_dict()

    def predict_on_pair(self, user: int, item: int):
        return self.user_means[user]


if __name__ == '__main__':
    train, validation = get_data()
    baseline_model = SimpleMean()
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
