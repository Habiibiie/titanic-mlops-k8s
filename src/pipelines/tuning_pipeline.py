import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Kendi modüllerimiz
from utils import read_params

# Path ayarı (src içinden classları bulabilmesi için)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.components.data_ingestion import load_data
from src.components.data_transformation import ColumnDropper, MissingValueImputer, CategoricalEncoder


def hyperparameter_optimization(config_path):
    config = read_params(config_path)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, config['external_data_config']['external_data_csv'])

    # 1. Veriyi Hazırla
    df = load_data(data_path)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Stratify ve Random State ayarlarına dikkat
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['preprocessing_config']['train_test_split_ratio'],
        random_state=config['preprocessing_config']['random_state'],
        stratify=y
    )

    # 2. Pipeline'ı Kur (Model hariç veya boş model ile)
    # GridSearch pipeline içindeki parametreleri denerken isimlendirmeye dikkat eder.
    # Format: "adımAdı__parametreAdı" (çift alt çizgi)
    pipeline = Pipeline([
        ('dropper', ColumnDropper(columns_to_drop=['PassengerId', 'Name', 'Ticket', 'Cabin'])),
        ('imputer', MissingValueImputer()),
        ('encoder', CategoricalEncoder()),
        ('model', RandomForestClassifier(random_state=42))
    ])

    # 3. Aranacak Parametre Izgarası (Grid)
    # Config dosyasından listeleri çekiyoruz
    param_grid = {
        'model__n_estimators': config['tuning_config']['n_estimators'],
        'model__max_depth': config['tuning_config']['max_depth']
    }

    print(f"🔍 Optimizasyon Başlıyor... Denenecek kombinasyonlar: {param_grid}")

    # 4. Grid Search Başlat
    # cv=3 -> 3 katlı çapraz doğrulama (Cross Validation) yapar
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    # 5. Sonuçları Göster
    print("\n-------------------------------------------")
    print(f"🏆 EN İYİ SKOR: {grid_search.best_score_:.4f}")
    print(f"🥇 EN İYİ PARAMETRELER: {grid_search.best_params_}")
    print("-------------------------------------------")
    print("💡 İPUCU: Bu değerleri params.yaml dosyasındaki 'model_config' kısmına yazabilirsin.")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'params.yaml')
    hyperparameter_optimization(config_path)