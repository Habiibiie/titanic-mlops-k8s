import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from utils import read_params

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.components.data_ingestion import load_data
from src.components.data_transformation import ColumnDropper, MissingValueImputer, CategoricalEncoder


def hyperparameter_optimization(config_path):
    config = read_params(config_path)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, config['external_data_config']['external_data_csv'])

    df = load_data(data_path)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['preprocessing_config']['train_test_split_ratio'],
        random_state=config['preprocessing_config']['random_state'],
        stratify=y
    )

    pipeline = Pipeline([
        ('dropper', ColumnDropper(columns_to_drop=['PassengerId', 'Name', 'Ticket', 'Cabin'])),
        ('imputer', MissingValueImputer()),
        ('encoder', CategoricalEncoder()),
        ('model', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'model__n_estimators': config['tuning_config']['n_estimators'],
        'model__max_depth': config['tuning_config']['max_depth']
    }

    print(f"🔍 Optimization begins... Combinations to be tried: {param_grid}")

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    print("\n-------------------------------------------")
    print(f"🏆 BEST SCORE: {grid_search.best_score_:.4f}")
    print(f"🥇 BEST PARAMETERS: {grid_search.best_params_}")
    print("-------------------------------------------")
    print("💡 TIP: You can add these values to the 'model_config' section of the params.yaml file.")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'params.yaml')
    hyperparameter_optimization(config_path)