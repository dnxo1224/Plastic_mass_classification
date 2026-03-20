import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sims_analyzer.processor import TOFSIMSAnalyzer


# 1. 설정 및 파일 분류
DATA_DIR = '../data/ASCII_v9 prod/'
MODE = 'POS'  # 'POS' 또는 'NEG' 선택
TARGET_RANGE = (2.0, 300.0)
BIN_WIDTH = 0.022
mass_bins = np.arange(TARGET_RANGE[0], TARGET_RANGE[1], BIN_WIDTH)

matrix_data = []
labels = []

# 파일 목록에서 특정 모드(NEG)만 필터링
file_list = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt') and MODE in f]

for file_name in file_list:
    label = file_name.split('_')[1]  # Product_HDPE_NEG... -> HDPE 추출
    analyzer = TOFSIMSAnalyzer(os.path.join(DATA_DIR, file_name))

    # 전처리 1단계: 정규화 (Normalization)
    analyzer.apply_preprocessing()  # 정규화 + 표준화
    indices = analyzer.find_peaks()

    # 공통 피크 리스트 생성 (Common Peak List)
    sample_vector = np.zeros(len(mass_bins))
    peaks_df = analyzer.df.iloc[indices]

    for _, row in peaks_df.iterrows():
        m_val = row['Mass (u)']
        if TARGET_RANGE[0] <= m_val < TARGET_RANGE[1]:
            bin_idx = int((m_val - TARGET_RANGE[0]) / BIN_WIDTH)
            if bin_idx < len(sample_vector):
                sample_vector[bin_idx] = row['Intensity']

    matrix_data.append(sample_vector)
    labels.append(label)

# 2. 통합 행렬 생성 및 전처리 2단계: 표준화 (Standardization)
X = pd.DataFrame(matrix_data, columns=[f"{m:.2f}" for m in mass_bins])
y = pd.Series(labels)

# 3. PCA 수행 및 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. PCA 결과를 데이터프레임으로 변환 (시각화용)
pca_df = pd.DataFrame(
    data=X_pca,
    columns=['PC1', 'PC2']
)

# 4. Loading 값 추출 (질문하신 코드의 시작점)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=X.columns
)

print("PCA 학습이 완료되었습니다.")
print(f"설명 가능한 분산 비율: {pca.explained_variance_ratio_}")

# Loading 값을 산점도로 표현
plt.figure(figsize=(10, 8))
plt.scatter(loadings['PC1'], loadings['PC2'], alpha=0.3, label='Other Peaks')

# 가중치가 큰 상위 10개 피크만 선과 라벨로 표시
top_features = loadings['PC1'].abs().sort_values(ascending=False).head(10).index

print(top_features)

for feature in top_features:
    plt.arrow(0, 0, loadings.loc[feature, 'PC1'], loadings.loc[feature, 'PC2'],
              color='red', alpha=0.8, head_width=0.02)
    plt.text(loadings.loc[feature, 'PC1']*1.2, loadings.loc[feature, 'PC2']*1.2,
             feature, color='blue', fontsize=12, weight='bold')


plt.axhline(0, color='black', linestyle='--')
plt.axvline(0, color='black', linestyle='--')
plt.xlabel(f'PC1 Loading ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 Loading ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('NEG PCA Loadings: Identifying Top 10 m/z Peaks')
plt.grid(True, alpha=0.3)
plt.show()