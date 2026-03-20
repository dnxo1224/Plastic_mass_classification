import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

def match_library(important_df, library, tolerance=0.1):
    matched_results = []
    for _, row in important_df.iterrows():
        exp_m = float(row['m/z'])
        importance = row['Importance']

        found = False
        for theory_m, info in library.items():
            if abs(exp_m - theory_m) <= tolerance:
                matched_results.append({
                    'Exp m/z': exp_m,
                    'Formula': info['Formula'],
                    'Assignment': info['Assignment'],
                    'Importance': importance
                })
                found = True
                break
        if not found:
            matched_results.append(
                {'Exp m/z': exp_m, 'Formula': 'Unknown', 'Assignment': 'N/A', 'Importance': importance})

    return pd.DataFrame(matched_results)


class TOFSIMSAnalyzer:
    def __init__(self, file_path):
        # 데이터 로드
        self.df = pd.read_csv(
            file_path,
            sep='\s+',
            skiprows=3,
            names=['Channel', 'Mass (u)', 'Intensity'],
        )
        self.df = self.df[self.df['Mass (u)'] >= 2].reset_index(drop=True)
        self.peaks = None

    # distance 565 ~= m/z 0.025
    def find_peaks(self, height=0.01, prominence=0.01, width=15, distance=565):
        """기본적인 피크 검출을 수행합니다."""
        indices, _ = find_peaks(self.df['Intensity'], height=height, prominence=prominence, width=width, distance=distance)
        self.peaks = indices
        return indices

    def normalize(self):
        """논문 방식: 각 피크 강도를 전체 신호 강도의 합으로 나눔 """
        total_sum = self.df['Intensity'].sum()
        if total_sum > 0:
            self.df['Intensity'] = self.df['Intensity'] / total_sum
        return self.df

    def apply_preprocessing(self):
        """논문 방식의 전처리를 수행합니다: 정규화 -> 표준화"""
        # 1. Normalization (Total Signal Intensity로 나누기)
        total_intensity = self.df['Intensity'].sum()
        self.df['Intensity'] = self.df['Intensity'] / total_intensity

        # 2. Standardization (StandardScaler 적용)
        scaler = StandardScaler()
        normalized_values = self.df['Intensity'].values.reshape(-1, 1)
        self.df['Intensity'] = scaler.fit_transform(normalized_values)

        return self.df

