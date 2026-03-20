import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from utils import Utils

class Models:

    def __init__(self):
        self.reg = {
            "SVR": SVR(),
            "GBR": GradientBoostingRegressor()
        }

        self.params = {
            "SVR": {
                "kernel": ["rbf", "linear"],
                "C": [1, 10, 100],
                "gamma": ["scale", "auto"]
            },
            "GBR": {
                "loss": ["squared_error", "absolute_error"],
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }
        }

    def grid_training(self, X, y):
        best_model = None
        best_score = 0
        for name, model in self.reg.items():
            grid_search = GridSearchCV(estimator=model, param_grid=self.params[name], cv=5)
            grid_search.fit(X, y)
            score = np.abs(grid_search.best_score_)

            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_
        
        utils = Utils()
        utils.model_exports(best_model, best_score)
        