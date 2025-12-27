import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Deletes the columns that defined.
    """
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        return X_copy.drop(self.columns_to_drop, axis=1, errors='ignore')

class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values.
    - Age: With mean.
    - Embarked: With mode.
    """
    def __init__(self):
        self.age_mean = None
        self.embarked_mode = None

    def fit(self, X, y=None):
        """
        Learns mean and mode from train data.
        """
        self.age_mean_ = X["Age"].mean()
        self.embarked_mode_ = X["Embarked"].mode()[0]
        print(f"Age mean that learned: {self.age_mean_}")
        print(f"Embarked mode that learned: {self.embarked_mode_}")
        return self

    def transform(self, X):
        """
        Writes the values that learned.
        """
        X_copy = X.copy()

        if self.age_mean_ is None or self.embarked_mode_ is None:
            raise RuntimeError("ERROR: You need to call fit() first.")

        X_copy["Age"] = X_copy["Age"].fillna(self.age_mean_)
        X_copy["Embarked"] = X_copy["Embarked"].fillna(self.embarked_mode_)
        return X_copy

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Turns categorical columns into numerical columns.
    """
    def __init__(self):
        self.sex_mapping = {'male': 0, 'female': 1}
        self.embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # --- DEBUG STARTING ----
        print(f"\n[DEBUG] Encoder Started.")
        print(f"[DEBUG] Embarked Type: {X_copy['Embarked'].dtype}")
        print(f"[DEBUG] Embarked first 5 rows: {X_copy['Embarked'].head(5).values}")
        print(f"[DEBUG] Dict that will use: {self.embarked_mapping}")
        # ---------------------

        # First: Gender Transform
        X_copy["Sex"] = X_copy["Sex"].map(self.sex_mapping).fillna(0).astype(int)

        # Second: Embarked Transform
        X_copy["Embarked"] = X_copy["Embarked"].map(self.embarked_mapping)

        X_copy["Embarked"] = pd.to_numeric(X_copy["Embarked"], errors='coerce').fillna(0).astype(int)

        # --- DEBUG RESULT ---
        print(f"[DEBUG] After Encoder Unique Embarked Values: {X_copy['Embarked'].unique()}")
        print("-" * 30)

        return X_copy

if __name__ == "__main__":
    df_test = pd.DataFrame({
        "PassengerId": [1,2,3],
        "Name" : ["Ali","Jenna","Veli"],
        "Age" : [20, np.nan, 40],
        "Sex" : ["male", "female", "male"],
        "Embarked" : ["S", np.nan, "C"],
    })

    print("--- Raw Data ---")
    print(df_test)

    # First Step: Dropping Columns
    dropper = ColumnDropper(columns_to_drop=["PassengerId", "Name"])
    df_dropped = dropper.transform(df_test)

    # Second Step: Imputing Missing Values
    imputer = MissingValueImputer()
    imputer.fit(df_dropped)
    df_imputed = imputer.transform(df_dropped)

    # Third Step: Encoding
    encoder = CategoricalEncoder()
    df_final = encoder.transform(df_imputed)

    print("\n--- Processed Data ---")
    print(df_final)