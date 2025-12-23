import os
import pickle
import sys
# YAML okuyucumuzu ekledik
from src.utils.common import read_params

# Path ayarı
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from src.components.data_ingestion import load_data
from src.components.data_transformation import ColumnDropper, MissingValueImputer, CategoricalEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_model(config_path):
    """
    Config dosyasından parametreleri okur ve eğitimi başlatır.
    """
    # 1. Parametreleri Yükle
    config = read_params(config_path)

    # Dosya yollarını config'den alıyoruz
    # (Proje ana dizinine göre yol oluşturuyoruz)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    data_path = os.path.join(base_dir, config['external_data_config']['external_data_csv'])
    model_dir = os.path.join(base_dir, config['model_config']['model_dir'])
    model_name = config['model_config']['model_name']
    model_path = os.path.join(model_dir, model_name)

    # Parametreleri alalım
    random_state = config['preprocessing_config']['random_state']
    split_ratio = config['preprocessing_config']['train_test_split_ratio']

    n_estimators = config['model_config']['n_estimators']
    max_depth = config['model_config']['max_depth']
    model_random_state = config['model_config']['random_state']

    # --- (Buradan sonrası aynı mantık, sadece değişkenleri kullanacağız) ---

    logger.info(f"Veri yükleniyor: {data_path}")
    df = load_data(data_path)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Config'den gelen split_ratio ve random_state kullanılıyor
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_ratio,
        random_state=random_state,
        stratify=y
    )

    pipeline = Pipeline([
        ('dropper', ColumnDropper(columns_to_drop=['PassengerId', 'Name', 'Ticket', 'Cabin'])),
        ('imputer', MissingValueImputer()),
        ('encoder', CategoricalEncoder()),
        # Config'den gelen parametreler burada!
        ('model', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=model_random_state
        ))
    ])

    logger.info(f"Model eğitiliyor. Parametreler: {n_estimators} ağaç")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model Accuracy Değeri: {accuracy}")

    logger.info(f"Model başarıyla şuraya kaydedildi: {model_path}")
    os.makedirs(model_dir, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    logger.info(f"Model Başarılı!")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "params.yaml")

    train_model(config_path)