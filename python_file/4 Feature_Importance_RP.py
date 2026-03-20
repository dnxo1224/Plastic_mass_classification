import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sims_analyzer.processor import match_library

ion_library = {
    52.01: {'Formula': 'C4H4-', 'Assignment': 'PS (Benzene fragment)'},
    103.05: {'Formula': 'C8H7-', 'Assignment': 'PS (Styrene fragment)'},
    72.99: {'Formula': 'C3H5O2-', 'Assignment': 'PET/Aliphatic oxide'},
    39.01: {'Formula': 'C3H3-', 'Assignment': 'Aliphatic (PP/HDPE)'},
    121.03: {'Formula': 'C7H5O2-', 'Assignment': 'PET (Benzoate)'},
    165.02: {'Formula': 'C8H5O4-', 'Assignment': 'PET (Terephthalate)'},
    35.00: {'Formula': 'Cl-', 'Assignment': 'PVC (Chlorine)'},
    45.99: {'Formula': 'C2H2O2-', 'Assignment': 'PET fragment'}
}


loaded_data = joblib.load('../matrix_data/processed_data_POS.pkl')
X_loaded = loaded_data['X']
y_loaded = loaded_data['y']

# 확인용
print(f"불러온 데이터 크기: {X_loaded.shape}")

# 1. 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_loaded, y_loaded, test_size=0.3, random_state=42, stratify=y_loaded)

# 2. Random Forest 모델 생성 및 학습
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 3. 모델 평가 (정확도 확인)
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 4. 특징 중요도(Feature Importance) 추출
importances = rf_model.feature_importances_
feature_names = X_loaded.columns # 질량(m/z) 값들
feature_importance_df = pd.DataFrame({'m/z': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

# 5. 중요도 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x='m/z', y='Importance', data=feature_importance_df, palette='magma', hue='m/z', legend=False)
plt.title('Top 10 Important m/z Peaks for POS Plastic Classification', fontsize=14)
plt.show()

final_report = match_library(feature_importance_df, ion_library)
print(final_report)
