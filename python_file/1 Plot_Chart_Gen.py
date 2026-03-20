# 12.23. 11:25 AM
# Author: Seolwootae

# 전처리 시각화 코드
import os
import matplotlib.pyplot as plt
from sims_analyzer.processor import TOFSIMSAnalyzer

file_path = '../data/ASCII_v9 prod/'
file_name = 'Product_PP_POS_Bi3_002_0.txt'
FILE_PATH = os.path.join(file_path, file_name)

# 전처리
analyzer = TOFSIMSAnalyzer(FILE_PATH)
df = analyzer.apply_preprocessing() # 정규화+표준화가 진행된 Intensity
indices = analyzer.find_peaks() # 전처리된 Intensity를 find_peaks.

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(df['Mass (u)'], df['Intensity'], color='black', linewidth=0.8)

# 정점 찍기
peak_masses = df['Mass (u)'].iloc[indices]
peak_intensities = df['Intensity'].iloc[indices]
plt.scatter(peak_masses, peak_intensities, color='red', marker='x', s=20, label='Detected Peaks')

# 그래프 꾸미기
plt.title(f'Mass Spectrum with Detected Peaks\n({file_name})', fontsize=14)
plt.xlabel('Mass (u)', fontsize=12)
plt.ylabel('Intensity', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

#분석 범위 제한
# plt.xlim(100, 110)
# plt.tight_layout()
plt.show()

