"""
model_factory.py

Contains all machine learning models and their 
hyperparameter grids used for training.
"""

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

from xgboost import XGBClassifier


RANDOM_STATE = 42


def get_models():
    """
    Returns dictionary of machine learning models.
    """

    models = {

        "DecisionTree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Bagging": BaggingClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "AdaBoost": AdaBoostClassifier(
            random_state=RANDOM_STATE
        ),

        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_estimators=100
        )

    }

    return models


def get_parameter_grid():
    """
    Returns parameter grids for GridSearchCV.
    """

    parameter_grid = {

        "DecisionTree": {

            "model__criterion": [
                "gini",
                "entropy"
            ],

            "model__max_depth": [
                3,
                5,
                7,
                10,
                None
            ],

            "model__min_samples_split": [
                2,
                5,
                10
            ]
        },


        "RandomForest": {

            "model__n_estimators": [
                100,
                200
            ],

            "model__max_depth": [
                5,
                10,
                None
            ],

            "model__min_samples_split": [
                2,
                5
            ]
        },


        "Bagging": {

            "model__n_estimators": [
                50,
                100,
                200
            ]
        },


        "AdaBoost": {

            "model__n_estimators": [
                50,
                100,
                200
            ],

            "model__learning_rate": [
                0.01,
                0.1,
                1.0
            ]
        },


        "GradientBoosting": {

            "model__n_estimators": [
                100,
                200
            ],

            "model__learning_rate": [
                0.01,
                0.1
            ],

            "model__max_depth": [
                3,
                5
            ]
        },


        "XGBoost": {

            "model__n_estimators": [
                100,
                200
            ],

            "model__learning_rate": [
                0.01,
                0.1
            ],

            "model__max_depth": [
                3,
                5,
                7
            ]
        }

    }

    return parameter_grid


if __name__ == "__main__":

    models = get_models()

    print("Available Models")

    print("=" * 40)

    for model_name in models.keys():

        print(model_name)