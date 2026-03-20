import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

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
plt.title('PCA of TOF-SIMS Peak-Picked Data', fontsize=15)
plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=12)
plt.legend(title='Plastic Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()