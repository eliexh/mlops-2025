import pandas as pd
from .base_preprocessor import BasePreprocessor

class TitanicPreprocessor:
    """Simple preprocessing for Titanic dataset"""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Cabin' in df.columns:
            df = df.drop(columns=['Cabin'])
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].mean())
        return df

