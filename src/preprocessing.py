"""
preprocessing.py 

Builds the preprocessing pipeline used during training
and deployment.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import pandas as pd


def identify_columns(df: pd.DataFrame):
    """
    Identify numerical and categorical columns.
    """

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return numerical_columns, categorical_columns


def build_preprocessor(df: pd.DataFrame):
    """
    Build preprocessing pipeline.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """

    numeric_columns, categorical_columns = identify_columns(df)

    ##########################################################
    # Numerical Pipeline
    ##########################################################

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    ##########################################################
    # Categorical Pipeline
    ##########################################################

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    ##########################################################
    # Column Transformer
    ##########################################################

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop"
    )

    return preprocessor


def get_feature_lists(df: pd.DataFrame):
    """
    Return feature names.

    Useful for debugging.
    """

    numeric_columns, categorical_columns = identify_columns(df)

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns
    }


if __name__ == "__main__":

    print("Preprocessing Module Loaded Successfully")