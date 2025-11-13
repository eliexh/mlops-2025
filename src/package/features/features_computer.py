import pandas as pd
from .base_features_computer import BaseFeaturesComputer

class TitanicFeaturesComputer(BaseFeaturesComputer):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Example feature
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        return df
