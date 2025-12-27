import os
import pickle
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

import mlflow
import mlflow.sklearn
from src.utils.common import read_params
from src.utils.logger import get_logger

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from src.components.data_ingestion import load_data
from src.components.data_transformation import ColumnDropper, MissingValueImputer, CategoricalEncoder

logger = get_logger(__name__)

# --- MLFLOW SETTINGS ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Titanic_Experiment")


def train_model(config_path):
    config = read_params(config_path)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    data_path = os.path.join(base_dir, config['external_data_config']['external_data_csv'])
    model_dir = os.path.join(base_dir, config['model_config']['model_dir'])
    model_name = config['model_config']['model_name']
    model_path = os.path.join(model_dir, model_name)

    random_state = config['preprocessing_config']['random_state']
    split_ratio = config['preprocessing_config']['train_test_split_ratio']

    n_estimators = config['model_config']['n_estimators']
    max_depth = config['model_config']['max_depth']
    model_random_state = config['model_config']['random_state']

    logger.info(f"Loading data: {data_path}")

    if not os.path.exists(data_path):
        logger.error(f"ERROR: Data file not found -> {data_path}")
        raise FileNotFoundError(f"{data_path} Not found. Please check the 'data/raw' folder.")

    df = load_data(data_path)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_ratio,
        random_state=random_state,
        stratify=y
    )

    # --- MLFLOW RUN ---
    with mlflow.start_run():
        logger.info("MLflow run has started... 🧪")

        pipeline = Pipeline([
            ('dropper', ColumnDropper(columns_to_drop=['PassengerId', 'Name', 'Ticket', 'Cabin'])),
            ('imputer', MissingValueImputer()),
            ('encoder', CategoricalEncoder()),
            ('model', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=model_random_state
            ))
        ])

        # MLflow Parameter Registration
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("split_ratio", split_ratio)
        mlflow.log_param("model_type", "RandomForestClassifier")

        logger.info(f"The model is being trained... (n_estimators={n_estimators}, max_depth={max_depth})")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy Value: {accuracy}")

        # MLflow Metric Logging
        mlflow.log_metric("accuracy", accuracy)

        # MLflow Model Registry (Save the model to the cloud/server)
        mlflow.sklearn.log_model(pipeline, "model")
        logger.info("The model has been saved to the MLflow database. 🚀")

        # --- LOCAL BACKUP (For API Use) ---
        logger.info(f"The model is being backed up to the local disk: {model_path}")
        os.makedirs(model_dir, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)

        logger.info(f"Pipeline completed successfully! ✅")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))

    config_path = os.path.join(project_root, "params.yaml")

    logger.info(f"Searching for configuration file: {config_path}")

    if not os.path.exists(config_path):
        logger.warning(f"WARNING: params.yaml not found at full path, trying default 'params.yaml'.")
        config_path = "params.yaml"

    train_model(config_path)