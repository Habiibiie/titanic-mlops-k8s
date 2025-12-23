# Hoca importları buraya attım, bazen çalışmıyor diye aşağı da ekledim.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veriyi oku (Dosya yolunu kendi bilgisayarımdaki gibi bıraktım, sen düzeltirsin abi)
df = pd.read_csv("data/raw/train.csv") # Kaggle'daki train.csv aslında bu

# Buralarda biraz veriye baktım
print(df.head())
print(df.columns)

# Gereksiz kolonları atalım dedik
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Eksik verileri doldurma (Burası çok önemli, ortalama ile doldurdum)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Kategorik dönüşümler (Elle yaptım, encoder kullanmayı beceremedim)
sex_mapping = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(sex_mapping)

embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
df['Embarked'] = df['Embarked'].map(embarked_mapping)

# Hedef ve öznitelikleri ayır
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train test diye bölelim
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli kuralım
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Sonuç ne çıktı?
print("Model Accuracy Değeri: ", accuracy_score(y_test, y_pred))

# Modeli kaydedeyim lazım olur
import pickle
pickle.dump(model, open('random_forest_model.pkl', 'wb'))

print("Model başarıyla kaydedildi kanka.")