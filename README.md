# 🧪 Plastic Mass Classification

TOF-SIMS(Time-of-Flight Secondary Ion Mass Spectrometry) 질량 스펙트럼 데이터를 활용한 **플라스틱 종류 자동 분류** 프로젝트입니다.

## 📋 프로젝트 개요

TOF-SIMS로 측정된 플라스틱 시료의 질량 스펙트럼 데이터를 전처리하고, 머신러닝 기법을 적용하여 **HDPE, PP, PET, PS** 4종의 플라스틱을 분류합니다.

### 분석 파이프라인

```
Raw ASCII Data → 전처리(정규화+표준화) → 피크 검출 → 질량 빈(Bin) 매핑 → PCA 시각화 → ML 분류 → Feature Importance 분석
```

## 📂 프로젝트 구조

```
plastic_mass_classification/
├── python_file/                    # 메인 분석 코드
│   ├── sims_analyzer/              # TOF-SIMS 전처리 모듈
│   │   ├── __init__.py
│   │   └── processor.py            # TOFSIMSAnalyzer 클래스 & 이온 라이브러리 매칭
│   ├── 1 Plot_Chart_Gen.py         # 질량 스펙트럼 시각화 및 피크 검출
│   ├── 2 PCA_loadings.py           # PCA Loading 분석 (주요 m/z 피크 식별)
│   ├── 2 classify_NEG_0.0235.py    # NEG 모드 PCA 분류 (bin=0.0235)
│   ├── 2 classify_NEG_0.032_mat.py # NEG 모드 PCA 분류 - Material 데이터
│   ├── 2 classify_POS_0.022.py     # POS 모드 PCA 분류 (bin=0.022)
│   ├── 2 classify_POS_HDPE_PP_0.02.py  # POS 모드 HDPE vs PP 분류
│   ├── 2 classify_POS_PET_PS_0.02.py   # POS 모드 PET vs PS 분류
│   ├── 3 matrix_X_data_save.py     # 전처리 데이터 행렬 저장 (pkl)
│   ├── 4 Feature_Importance_ET.py  # Extra Trees 기반 Feature Importance
│   ├── 4 Feature_Importance_RP.py  # Random Forest 기반 Feature Importance
│   └── 5 Feature_Importance_SHAP.py # SHAP 기반 Feature Importance
├── matrix_data/                    # 전처리된 데이터 행렬 (pkl, gitignore 대상)
├── old_code/                       # 초기 개발 코드 (레거시)
│   ├── MofZ.py                     # m/z 스펙트럼 시각화 (초기 버전)
│   ├── make_matrix.py              # 데이터 행렬 생성 (초기 버전)
│   ├── mass.py                     # 질량 스펙트럼 분석 (초기 버전)
│   └── visualize_pca.py            # PCA 시각화 (초기 버전)
└── .gitignore
```

## 🔬 핵심 모듈

### `sims_analyzer/processor.py`

| 클래스/함수 | 설명 |
|---|---|
| `TOFSIMSAnalyzer` | TOF-SIMS ASCII 데이터 로드, 전처리(정규화/표준화), 피크 검출을 수행하는 핵심 클래스 |
| `match_library()` | 검출된 m/z 피크를 이온 라이브러리와 매칭하여 화학적 해석을 지원 |

### 전처리 과정

1. **정규화(Normalization)**: 각 피크 강도를 전체 신호 강도의 합으로 나눔
2. **표준화(Standardization)**: `StandardScaler`를 적용하여 평균 0, 분산 1로 스케일링
3. **피크 검출**: `scipy.signal.find_peaks`를 사용하여 주요 질량 피크를 검출
4. **질량 빈 매핑**: 검출된 피크를 공통 질량 범위(2.0~300.0 m/z)의 균일 빈에 매핑

## 🚀 사용법

### 1. 환경 설정

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy joblib shap
```

### 2. 분석 실행 순서

```bash
# 1단계: 스펙트럼 시각화 및 확인
python "python_file/1 Plot_Chart_Gen.py"

# 2단계: PCA 분류 및 시각화
python "python_file/2 classify_POS_0.022.py"

# 3단계: 전처리 데이터 저장
python "python_file/3 matrix_X_data_save.py"

# 4단계: Feature Importance 분석
python "python_file/4 Feature_Importance_ET.py"
python "python_file/4 Feature_Importance_RP.py"

# 5단계: SHAP 분석
python "python_file/5 Feature_Importance_SHAP.py"
```

## 🛠️ 기술 스택

| 분류 | 라이브러리 |
|---|---|
| 데이터 처리 | `numpy`, `pandas` |
| 전처리 | `scipy.signal`, `sklearn.preprocessing` |
| 차원 축소 | `sklearn.decomposition.PCA` |
| 분류 모델 | `RandomForestClassifier`, `ExtraTreesClassifier` |
| 해석 가능성 | `SHAP (TreeExplainer)` |
| 시각화 | `matplotlib`, `seaborn` |

## 📊 분석 모드

- **POS 모드**: 양이온 질량 스펙트럼 분석
- **NEG 모드**: 음이온 질량 스펙트럼 분석
- **Material 데이터**: 원재료(Material) 시료 분석

## 👤 Author

**Seolwootae**
