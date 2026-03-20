import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import shap

plastic_names = ['HDPE', 'PET', 'PS', 'PP']

loaded_data = joblib.load('../matrix_data/processed_data_NEG.pkl')
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
# plt.figure(figsize=(10, 6))
# sns.barplot(x='m/z', y='Importance', data=feature_importance_df, palette='magma', hue='m/z', legend=False)
# plt.title('Top 10 Important m/z Peaks for POS Plastic Classification', fontsize=14)
# plt.show()

# --- 6. SHAP 분석 추가 ---
# Tree 기반 모델 전용 Explainer를 생성합니다.
explainer = shap.TreeExplainer(rf_model)

# 테스트 데이터(X_test)에 대한 SHAP value를 계산합니다.
# 이 과정은 모델이 각 샘플을 분류할 때 각 m/z가 미친 영향력을 수치화합니다.
shap_values = explainer(X_test)

# SHAP Summary Plot 시각화
# 각 클래스별로 어떤 피크가 결정적이었는지 한눈에 보여줍니다.
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X_test.columns, class_names=plastic_names)

# 특정 클래스(예: HDPE)에 대해 피크의 강도가 어떻게 영향을 주었는지 보고 싶을 때
# shap_values[index]의 index는 모델의 클래스 순서에 따릅니다.
# shap.summary_plot(shap_values[0], X_test)
