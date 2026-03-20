import os
import pandas as pd
import numpy as np
from sims_analyzer.processor import TOFSIMSAnalyzer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# 1. 설정
DATA_DIR = '../data/ASCII_v9 prod/'
TARGET_RANGE = (2.0, 300.0)  # 논문에서 사용한 m/z 범위 [cite: 368]
BIN_WIDTH = 0.02  # 논문에서 최적화한 질량 간격

# 공통 질량 빈(Bin) 생성
mass_bins = np.arange(TARGET_RANGE[0], TARGET_RANGE[1], BIN_WIDTH)
matrix_data = []
labels = []

# 2. 폴더 내 파일 순회
file_list = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]

print(f"총 {len(file_list)}개의 파일을 처리합니다...")

for file_name in file_list:
    # 파일 경로 및 라벨 추출 (파일명에서 플라스틱 종류 추출)
    # 예: Product_HDPE_... -> HDPE
    label = file_name.split('_')[1]
    full_path = os.path.join(DATA_DIR, file_name)

    # Analyzer 객체 생성 및 전처리
    analyzer = TOFSIMSAnalyzer(full_path)
    analyzer.apply_preprocessing()  # 정규화 + 표준화
    indices = analyzer.find_peaks()  # 최적 하이퍼파라미터 적용 [cite: 366, 752]

    # 검출된 피크 데이터 추출
    peaks_df = analyzer.df.iloc[indices]

    # 공통 빈(Bin)에 데이터 매핑
    sample_vector = np.zeros(len(mass_bins))
    for _, row in peaks_df.iterrows():
        m_val = row['Mass (u)']
        if TARGET_RANGE[0] <= m_val < TARGET_RANGE[1]:
            # 가장 가까운 빈 인덱스 찾기
            bin_idx = int((m_val - TARGET_RANGE[0]) / BIN_WIDTH)
            if bin_idx < len(sample_vector):
                # 해당 빈에 표준화된 강도 값 저장
                sample_vector[bin_idx] = row['Intensity']

    matrix_data.append(sample_vector)
    labels.append(label)

# 3. 통합 데이터프레임 생성
# 행: 샘플, 열: 질량 빈(Mass Bins)
X = pd.DataFrame(matrix_data, columns=[f"{m:.2f}" for m in mass_bins])
y = pd.Series(labels)

print("통합 행렬 생성 완료!")
print(f"행렬 크기: {X.shape} (샘플 수 x 피크 특징 수)")


# 1. PCA 수행
# n_components=2: 가장 분산이 큰 2개의 주성분(PC1, PC2)을 추출합니다.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 2. 주성분이 설명하는 분산 비율 확인
# 모델이 원본 데이터의 정보를 얼마나 잘 보존하고 있는지 나타냅니다.
explained_variance = pca.explained_variance_ratio_

# 3. 데이터프레임으로 정리
pca_df = pd.DataFrame(
    data=X_pca,
    columns=['Principal Component 1', 'Principal Component 2']
)
pca_df['Label'] = y

# 4. 시각화
plt.figure(figsize=(10, 8))
# 플라스틱 종류별로 색상을 다르게 설정합니다.
sns.scatterplot(
    x='Principal Component 1',
    y='Principal Component 2',
    hue='Label',
    style='Label',
    data=pca_df,
    s=100,
    palette='viridis'
)

# 그래프 정보 추가
plt.title(f'PCA of TOF-SIMS Peak-Picked Data', fontsize=15)
plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=12)
plt.legend(title='Plastic Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()