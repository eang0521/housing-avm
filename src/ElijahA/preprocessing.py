from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# Insert Transformer code below
class TypeDummyCreator(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns if columns is not None else ['type']

    def fit(self, X, y=None):
        self.categories_ = {
            col: sorted(X[col].dropna().unique().tolist())
            for col in self.columns
        }
        return self

    def transform(self, X):
        X_out = X.copy().reset_index(drop=True)
        for col in self.columns:
            dummies = pd.get_dummies(X_out[col], prefix=col).astype(int)
            expected = [f"{col}_{cat}" for cat in self.categories_[col]]
            dummies = dummies.reindex(columns=expected, fill_value=0)
            X_out = pd.concat([X_out.drop(columns=[col]), dummies], axis=1)
        return X_out

class CustomCategoricalEncoder(BaseEstimator, TransformerMixin):
    """A custom transformer for encoding categorical variables."""
    
    def __init__(self):
        self.columns = None

    def fit(self, X: pd.DataFrame, y=None):
        """Learn the unique categories from the data."""
        self.columns = X.select_dtypes(include=['object']).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns to numeric using one-hot encoding."""
        return pd.get_dummies(X, columns=self.columns, drop_first=True)


class CustomNumericImputer(BaseEstimator, TransformerMixin):
    """A custom transformer for imputing missing numerical values."""
    
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imputed_values_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Compute the imputed values based on the strategy."""
        for col in X.select_dtypes(include=['float64', 'int64']).columns:
            if self.strategy == 'mean':
                self.imputed_values_[col] = X[col].mean()
            elif self.strategy == 'median':
                self.imputed_values_[col] = X[col].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in numeric columns."""
        for col, value in self.imputed_values_.items():
            X[col] = X[col].fillna(value)
        return X

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset for model training.
    - Imputes missing unemployment values with the column median.
    - One-hot encodes 'type' and 'city' via TypeDummyCreator.
    Returns a DataFrame with numeric columns only (sold_price included as target).
    """
    df = data.copy()

    num_features = ['bedrooms', 'sq_ft', 'build_age', 'school_score', 'unemployment', 'interest_rate']
    cat_features = ['type', 'city']

    imputer = CustomNumericImputer(strategy='median')
    df[num_features] = imputer.fit_transform(df[num_features])

    encoder = TypeDummyCreator(columns=cat_features)
    df = encoder.fit_transform(df)

    return df
