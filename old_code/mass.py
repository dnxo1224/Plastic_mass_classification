import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

file_path = '../data/plastic_mini/'
file_name = 'Product_HDPE_NEG_Bi3_001_0.txt'
df = pd.read_csv(file_path+file_name, sep='\s+', skiprows=3, names=['Channel', 'Mass(u)', 'Intensity'])

print("Column names:", df.columns.tolist())

peaks, properties = find_peaks(df['Intensity'], height=1500, distance=1000)

peak_masses = df['Mass(u)'].iloc[peaks]
peak_intensities = df['Intensity'].iloc[peaks]

pp_library = {
        1.01: "H+", 12.00: "C+", 15.02: "CH3+", 27.02: "C2H3+",
        29.04: "C2H5+", 41.04: "C3H5+", 43.06: "C3H7+",
        55.05: "C4H7+", 57.07: "C4H9+"
    }

plt.figure(figsize=(15, 6))
plt.plot(df['Mass(u)'], df['Intensity'], color='black', linewidth=0.8)
plt.scatter(peak_masses, peak_intensities, color='red', s=10, label='Detected Peaks')

for m, i in zip(peak_masses, peak_intensities):
    found = False
    for lib_m, name in pp_library.items():
        # 라이브러리 질량과 오차범위(0.05) 내에 있는지 확인
        if abs(m - lib_m) < 0.05:
            plt.annotate(name, xy=(m, i), xytext=(0, 10),
                         textcoords='offset points', ha='center',
                         fontsize=9, fontweight='bold', color='blue',
                         arrowprops=dict(arrowstyle='->', color='blue', alpha=0.3))
            found = True
            break

    # 라이브러리에 없는 피크인 경우 질량(Mass) 값 표시
    if not found:
        plt.annotate(f"{m:.2f}", xy=(m, i), xytext=(0, 10),
                     textcoords='offset points', ha='center',
                     fontsize=8, color='darkgreen',
                     arrowprops=dict(arrowstyle='->', color='green', alpha=0.2))

# 그래프 설정
plt.title(f'TOF-SIMS Mass Spectrum - {file_name}', fontsize=15)
plt.xlabel('Mass(u)', fontsize=12)
plt.ylabel('Intensity', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
# plt.xlim(37.5, 47.5)  # 보고 싶은 질량 범위 설정

plt.tight_layout()
plt.show()
