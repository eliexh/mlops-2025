from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for models."""

    @abstractmethod
    def train(self, X, y):
        """Train the model."""
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """Evaluate the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass
