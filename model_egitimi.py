import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

# 1. VERİYİ YÜKLE
print("⏳ Veri yükleniyor...")
columns = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings",
    "EmploymentSince", "InstallmentRate", "PersonalStatusSex", "OtherDebtors",
    "PresentResidenceSince", "Property", "Age", "OtherInstallmentPlans", "Housing",
    "NumberExistingCredits", "Job", "NumberPeopleMaintenance", "Telephone",
    "ForeignWorker", "Target"
]

# Dosya adınızın 'german.data' olduğundan emin olun
df = pd.read_csv('german.data', sep='\s+', header=None, names=columns)

# Hedef değişkeni ayarla (1: İyi, 2: Kötü -> 0: Kötü, 1: İyi)
df['Target_bin'] = (df['Target'] == 1).astype(int)

# 2. ÖZELLİKLERİ BELİRLE
numeric_cols = ["Duration", "CreditAmount", "InstallmentRate", "PresentResidenceSince",
                "Age", "NumberExistingCredits", "NumberPeopleMaintenance"]

# Hedef ve Target_bin hariç diğerleri kategorik
categorical_cols = [c for c in df.columns if c not in numeric_cols + ['Target', 'Target_bin']]

final_features = numeric_cols + categorical_cols

X = df[final_features]
y = df['Target_bin']

# 3. ÖN İŞLEME VE MODEL (PIPELINE)
# Bu kısım app.py ile uyumlu olmalı
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Modeli oluştur
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Pipeline oluştur (Önce işle, sonra eğit)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

# 4. EĞİTİM
print("⚙️ Model eğitiliyor...")
clf.fit(X, y)

# 5. KAYDETME (AYRI AYRI)
# app.py'nin beklediği yapıya göre parçalara ayırıp kaydediyoruz
print("💾 Dosyalar kaydediliyor...")

# a) Sadece eğitilmiş modeli (classifier adımını) kaydet
final_model = clf.named_steps['classifier']
joblib.dump(final_model, 'final_model.pkl')

# b) Sadece ön işlemciyi (preprocessor adımını) kaydet
final_preprocessor = clf.named_steps['preprocessor']
joblib.dump(final_preprocessor, 'preprocessor.pkl')

# c) Özellik listesini kaydet
joblib.dump(final_features, 'final_features.pkl')

print("✅ İŞLEM TAMAMLANDI! Yeni .pkl dosyaları oluşturuldu.")