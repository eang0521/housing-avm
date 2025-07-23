from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# Insert Transformer code below
class TypeDummyCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Fit logic, if needed, goes here
        return self

    def transform(self, X):
        # Transform logic: Assume X is a DataFrame
        # For example, let's say we want to scale the "feature_1" column
        X_transformed = X.copy().reset_index().drop("index", axis=1)
        dummy_df = pd.get_dummies(X_transformed["type"]).reset_index().drop("index", axis=1).astype("int")
        X_transformed_out = pd.concat([
            X_transformed[[x for x in X_transformed.columns if x != "type"]],
            dummy_df,
        ], axis=1)
        return X_transformed_out

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
    """Preprocess the dataset using custom transformers."""
    # Define features
    num_features = ['sold_price', 'bedrooms', 'sq_ft', 'build_age', 'school_score', 'unemployment', 'interest_rate']
    cat_features = ['address', 'city', 'type']

    # Create a column transformer with custom transformers
    preprocess = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', CustomNumericImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', CustomCategoricalEncoder(), cat_features)
        ],
        remainder='passthrough'  # Keep columns that are not specified
    )
    
    # Transform the data
    processed_data = preprocess.fit_transform(data)
    
    # Return processed DataFrame
    # If needed, convert the transformed output to a DataFrame
    return pd.DataFrame(processed_data, columns=num_features + list(preprocess.named_transformers_['cat'].columns))
