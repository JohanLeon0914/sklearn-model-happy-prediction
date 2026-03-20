import pandas as pd
import joblib

class Utils:
    def __init__(self):
        pass

    def load_from_csv(self, path):
        return pd.read_csv(path)

    def features_target(self, data, drop_columns, target_column):
        X = data.drop(columns=drop_columns)
        y = data[target_column]
        return X, y

    def model_exports(self, clf, score):
        print(f"Best Model: {clf}, Score: {score}")
        joblib.dump(clf, "./out/best_model.pkl")