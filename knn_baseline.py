import numpy as np
import pandas as pd
import pickle
from linear_regression_baseline import Baseline

from config import BASELINE_PARAMS_FILE_PATH, ITEM_COL_NAME_IN_DATASET, RATING_COL_NAME_IN_DATASET, USER_COL_NAME_IN_DATAEST,CORRELATION_PARAMS_FILE_PATH
from interface import Regressor
from utils import get_data, Config

class KnnBaseline(Regressor):
    def __init__(self, config):
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        self.corr_csv = None
        self.user_map = None
        self.item_map = None
        self.data = None
        self.k = config.k

    def fit(self, X: np.array):
        self.upload_params()
        self.data = X

    def predict_on_pair(self, user: int, item: int):
        try:
            user_input_item_bias = self.global_mean + self.user_biases[user] + self.item_biases[item]
            user_rated_items = self.data.loc[self.data[USER_COL_NAME_IN_DATAEST]==user].set_index(ITEM_COL_NAME_IN_DATASET, drop=False)
            user_rated_items['User_Rated_Items_Bias'] = pd.DataFrame.from_dict({id : self.item_biases[id]+ self.global_mean + self.user_biases[user] for id in user_rated_items[ITEM_COL_NAME_IN_DATASET]},orient='index')
            input_item_similarities = self.get_input_item_similarities(item)
            joined = user_rated_items.join(input_item_similarities, how='inner').drop(['item'], axis=1).sort_values(by='sim',ascending=False)[0:self.k]
            predicted_rating = user_input_item_bias + (sum((joined[RATING_COL_NAME_IN_DATASET]-joined['User_Rated_Items_Bias'])*joined['sim']))/sum(joined['sim'])
            return np.clip(predicted_rating,1,5)
        except:
            if user in self.data[USER_COL_NAME_IN_DATAEST]:
                predicted_rating = self.data.loc[self.data[USER_COL_NAME_IN_DATAEST] == user][RATING_COL_NAME_IN_DATASET].mean()
            else:
                predicted_rating = self.global_mean
        return predicted_rating

    def upload_params(self):
        self.corr_csv = pd.read_csv(CORRELATION_PARAMS_FILE_PATH, header=0)
        loaded_model = pickle.load(open(BASELINE_PARAMS_FILE_PATH, 'rb'))
        self.user_biases = loaded_model.user_biases
        self.item_biases = loaded_model.item_biases
        self.global_mean = loaded_model.global_mean

    def get_input_item_similarities(self, item):
        input_item_similarities = self.corr_csv[self.corr_csv['item_1'] == item].set_index('item_2', drop=False).reindex(columns=['item_2','sim'])
        input_item_similarities.rename(columns={'item_2': 'item'}, inplace=True)
        input_item_similarities2 = self.corr_csv[self.corr_csv['item_2'] == item].set_index('item_1', drop=False).reindex(columns=['item_1','sim'])
        input_item_similarities2.rename(columns={'item_1': 'item'}, inplace=True)
        input_item_similarities_full = pd.concat([input_item_similarities, input_item_similarities2])
        input_item_similarities_full.drop_duplicates(inplace=True)
        return input_item_similarities_full


if __name__ == '__main__':
    baseline_knn_config = Config(k=25)
    train, validation = get_data()
    knn_baseline = KnnBaseline(baseline_knn_config)
    knn_baseline.fit(train)
    print(knn_baseline.calculate_rmse(validation))
