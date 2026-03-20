import os
import pandas as pd
import numpy as np
import joblib # 머신러닝 데이터 저장에 최적화된 라이브러리
from sims_analyzer.processor import TOFSIMSAnalyzer

# 1. 설정 및 파일 분류
DATA_DIR = '../data/ASCII_v9 prod/'
MODE = 'POS'  # 'POS' 또는 'NEG' 선택
TARGET_RANGE = (2.0, 300.0)
BIN_WIDTH = 0.022
mass_bins = np.arange(TARGET_RANGE[0], TARGET_RANGE[1], BIN_WIDTH)

matrix_data = []
labels = []

# 파일 목록에서 특정 모드만 필터링
file_list = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt') and MODE in f]

# 파일 목록에서 특정 모드의 HDPE와 PP만 필터링
# file_list = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt') and MODE in f and ('PET' in f or 'PS' in f)]

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


# 저장할 폴더 생성
save_dir = '../matrix_data/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 2. Joblib 형식으로 저장 (머신러닝 학습 시 매우 빠르고 정확함)
# X와 y를 딕셔너리 형태로 묶어 파일 하나로 저장합니다.
data_to_save = {
    'X': X,
    'y': y,
    'mass_bins': mass_bins
}
joblib_path = os.path.join(save_dir, f'processed_data_{MODE}.pkl')
joblib.dump(data_to_save, joblib_path)

print(f"--- 저장 완료 ---")
print(f"Binary 저장 위치: {joblib_path}")