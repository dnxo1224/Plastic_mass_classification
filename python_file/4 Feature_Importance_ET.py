from sklearn.ensemble import ExtraTreesClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

loaded_data = joblib.load('../matrix_data/processed_data_POS.pkl')
X_loaded = loaded_data['X']
y_loaded = loaded_data['y']

# 확인용
print(f"불러온 데이터 크기: {X_loaded.shape}")

# 1. 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_loaded, y_loaded, test_size=0.3, random_state=42, stratify=y_loaded)

# 2. Random Forest 모델 생성 및 학습
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_model.fit(X_train, y_train)

# 3. 모델 평가
y_pred = et_model.predict(X_test)
print("--- Extra Trees Classification Report ---")
print(classification_report(y_test, y_pred))

# 4. 특징 중요도 추출 및 시각화
et_importances = et_model.feature_importances_
et_feature_importance_df = pd.DataFrame({
    'm/z': X_loaded.columns,
    'Importance': et_importances
}).sort_values(by='Importance', ascending=False).head(10)

# 5. 중요도 시각화
plt.figure(figsize=(10, 6))
sns.barplot(
    x='m/z', y='Importance', data=et_feature_importance_df,
    palette='viridis', hue='m/z', legend=False
)
plt.title('Top 10 Important m/z Peaks for POS (Extra Trees)', fontsize=14)
plt.show()