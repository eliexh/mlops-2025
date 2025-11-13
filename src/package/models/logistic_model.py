import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .base_model import BaseModel

class LogisticModel(BaseModel):
    """Logistic Regression implementation of BaseModel."""

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        preds = self.model.predict(X)
        return accuracy_score(y, preds)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
