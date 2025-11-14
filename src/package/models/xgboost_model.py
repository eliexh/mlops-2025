from xgboost import XGBClassifier
import pickle
from sklearn.metrics import accuracy_score

class XGBoostModel:
    def __init__(self, **kwargs):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
