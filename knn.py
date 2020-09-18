import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import csv

from config import CSV_COLUMN_NAMES, CORRELATION_PARAMS_FILE_PATH, \
    USER_COL_NAME_IN_DATAEST, ITEM_COL_NAME_IN_DATASET, RATING_COL_NAME_IN_DATASET, ITEMS_COL_INDEX, USERS_COL_INDEX, RATINGS_COL_INDEX
from interface import Regressor
from utils import get_data, Config


class KnnItemSimilarity(Regressor):
    def __init__(self, config):
        self.k = config.k
        self.n_users = None
        self.n_items = None
        self.data = None
        self.user_map = None
        self.corr_matrix = None
        self.corr_csv = None
        self.items_dict = {}
        self.items_mean_rating = {}

    def fit(self, X: np.array):
        self.data = X
        self.user_map = pd.Series(X[USER_COL_NAME_IN_DATAEST].unique())
        if not os.path.isfile(CORRELATION_PARAMS_FILE_PATH):
            for row in X.to_numpy():
                if(row[ITEMS_COL_INDEX] not in self.items_dict.keys()):
                    self.items_dict[row[ITEMS_COL_INDEX]] = {}
                self.items_dict[np.int16(row[ITEMS_COL_INDEX])][np.int16(row[USERS_COL_INDEX])] = np.int16(row[RATINGS_COL_INDEX])
            self.build_item_to_itm_corr(train)
            self.save_params(self.corr_matrix)
        self.upload_params()

    def predict_on_pair(self, user, item):
        try:
            user_rated_items = self.data.loc[self.data[USER_COL_NAME_IN_DATAEST] == user].set_index(ITEM_COL_NAME_IN_DATASET, drop=False) #df.rename({'a': 'X', 'b': 'Y'}, axis=1
            input_item_similarities = self.get_input_item_similarities(item)
            joined = user_rated_items.join(input_item_similarities, how='inner').drop(['item'], axis=1).sort_values(by='sim',ascending=False)[0:self.k]
            predicted_rating = sum(joined[RATING_COL_NAME_IN_DATASET]*joined['sim'])/sum(joined['sim'])
            return np.clip(predicted_rating, 1, 5)
        except:
            if user in self.data[USER_COL_NAME_IN_DATAEST]:
                predicted_rating = self.data.loc[self.data[USER_COL_NAME_IN_DATAEST] == user][RATING_COL_NAME_IN_DATASET].mean()
            else:
                predicted_rating = self.data[RATING_COL_NAME_IN_DATASET].mean()
            return predicted_rating

    def upload_params(self):
        self.corr_csv = pd.read_csv(CORRELATION_PARAMS_FILE_PATH, header=0)

    def save_params(self, matrix):
        with open(CORRELATION_PARAMS_FILE_PATH, 'w') as csv_file:
            writer = csv.writer(csv_file,lineterminator='\n')
            writer.writerow(CSV_COLUMN_NAMES)
            counter = 0
            for first_item, row in tqdm(matrix.iterrows()):
                counter += 1
                for second_item, sim in (row[-(row.size-counter):]).items():
                    if not(first_item == second_item):
                        intersection_group = set(self.items_dict[first_item].keys()).intersection(set(self.items_dict[second_item].keys()))
                        if len(intersection_group) > 1:
                            if (sim > 0):
                                writer.writerow([np.int16(first_item),np.int16(second_item),np.float32(round(sim, 6))])

    def build_item_to_itm_corr(self, X):
        self.user_map = pd.Series(X[USER_COL_NAME_IN_DATAEST].unique())
        df_movie_features = X.pivot(index=ITEM_COL_NAME_IN_DATASET, columns=USER_COL_NAME_IN_DATAEST, values=RATING_COL_NAME_IN_DATASET).fillna(0)
        self.item_map = pd.Series(df_movie_features.index)
        self.corr_matrix = np.corrcoef(df_movie_features)
        self.corr_matrix = pd.DataFrame(self.corr_matrix).set_index(self.item_map,self.item_map)
        self.corr_matrix.columns = [self.item_map][0].values


    def get_input_item_similarities(self, item):
        input_item_similarities = self.corr_csv[self.corr_csv['item_1'] == item].set_index('item_2', drop=False).reindex(columns=['item_2','sim'])
        input_item_similarities.rename(columns={'item_2': 'item'}, inplace=True)
        input_item_similarities2 = self.corr_csv[self.corr_csv['item_2'] == item].set_index('item_1', drop=False).reindex(columns=['item_1','sim'])
        input_item_similarities2.rename(columns={'item_1': 'item'}, inplace=True)
        input_item_similarities_full = pd.concat([input_item_similarities, input_item_similarities2])
        input_item_similarities_full.drop_duplicates(inplace=True)
        return input_item_similarities_full


if __name__ == '__main__':
    knn_config = Config(k=25)
    train, validation = get_data()
    knn = KnnItemSimilarity(knn_config)
    knn.fit(train)
    print(knn.calculate_rmse(validation))
