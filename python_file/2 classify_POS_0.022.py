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

pca_df = pd.DataFrame(X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Type'] = y

plt.figure(figsize=(10, 7))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Type', style='Type', data=pca_df, s=100, palette='viridis')
plt.title(f'PCA of TOF-SIMS {MODE} Peak-Picked Data', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)',fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)',fontsize=12)
plt.legend(title='Plastic Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
