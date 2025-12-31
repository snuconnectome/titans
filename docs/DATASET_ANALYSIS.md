# 📄 Grand Budapest Hotel fMRI Dataset Analysis

## 1. 📚 Paper Info
- **Title**: An fMRI dataset in response to "The Grand Budapest Hotel", a socially-rich, naturalistic movie
- **Authors**: Matteo Visconti di Oleggio Castello et al.
- **Journal**: Scientific Data (Nature), 2020
- **Dataset ID**: OpenNeuro ds003017

---

## 2. 🧪 Deep Dive: Proposed Benchmark Tasks

본 프로젝트(Titans-Neuro)에서는 단순한 성능 측정을 넘어, 모델이 **뇌의 동적 특성(Dynamics)**과 **영화의 서사(Narrative Context)**를 실제로 이해했는지 검증하기 위해 심화된 태스크를 정의합니다.

### ✅ Task 1: Multi-step Brain Trajectory Prediction (Dynamics)
> **"Specific Goal: Predict next 3 volumes (6 seconds) via Auto-regressive Rollout"**

- **문제 정의**: $x_{0:t}$가 주어졌을 때, 미래의 연속된 시퀀스 $x_{t+1}, x_{t+2}, x_{t+3}$을 예측합니다.
- **Horizon 결정 ($k=3$, 6초)의 과학적 근거**:
  1.  **HRF (Hemodynamic Response Function) Delay**: 
      - 신경 세포가 자극을 받으면 혈류량(fMRI 신호)이 피크에 도달하기까지 **약 4~6초**가 걸립니다.
      - $t+1$ (2초): 반응 시작 (Onset)
      - $t+2$ (4초): 상승기 (Rise)
      - $t+3$ (6초): 피크 도달 (Peak)
      - 따라서 **3 Step**을 예측해야만 "자극(원인) -> 뇌 반응(결과)"의 전체 인과 과정을 모델링했다고 볼 수 있습니다.
  2.  **Trade-off**: 
      - 6초 이후(Undershoot)는 예측 난이도가 급격히 상승하며, Auto-regressive 에러 누적으로 인해 검증 신뢰도가 떨어집니다. $k=3$이 최적의 균형점입니다.

- **구현 방식**: **Auto-regressive Rollout**
  1. $\hat{x}_{t+1} = Model(x_{0:t})$
  2. $\hat{x}_{t+2} = Model(x_{0:t}, \hat{x}_{t+1})$ (자신의 예측을 입력으로 재사용)
  3. $\hat{x}_{t+3} = Model(x_{0:t}, \hat{x}_{t+1}, \hat{x}_{t+2})$

- **평가 지표**: Average Voxel Correlation over 3 steps

### ✅ Task 2: Brain Encoding (Regression)
> "영화의 시청각 정보를 뇌 활동 신호로 변환할 수 있는가?"

- **문제 정의**: 영화의 특징(Feature) $S_t$가 주어졌을 때, 뇌 활동 $B_t$를 예측합니다. (Stimulus $\to$ Brain)
- **Type**: **Regression (회귀)**
  - 뇌 활동은 연속적인 실수값(Continuous Real Values)입니다.
- **입력**: 
  - Visual Features (ResNet, CLIP), Audio Features (Mel-spectrogram), Semantic Features (BERT)
- **출력**: 
  - Whole Brain Volume or ROI (Region of Interest) Voxels
- **핵심**: Titans Memory가 영화의 **맥락(Context)**을 얼마나 잘 압축하여 저장하고 있다가, 혈류역학적 지연(Hemodynamic Delay, 약 4~6초)을 고려해 뇌 신호로 변환해내는지 봅니다.

### ✅ Task 3: Scene/Context Decoding (Classification)
> "뇌 활동만 보고 지금 어떤 장면을 보고 있는지 맞출 수 있는가?"

- **문제 정의**: 뇌 활동 $B_{t-w:t}$가 주어졌을 때, 현재 영화 장면의 라벨 $Y_t$를 예측합니다. (Brain $\to$ Stimulus Class)
- **Type**: **Multi-class Classification (분류)**
- **라벨 예시**: 
  - Scene IDs (Scene 1, Scene 2...)
  - Context Tags (Face, Indoor, Dialogue, Action...)
- **평가 지표**: Accuracy, F1-Score

---

## 3. 📂 Folder Structure
```
titans/
├── papers/
│   └── GrandBudapest_fMRI_NatureData.pdf
├── docs/
│   └── DATASET_ANALYSIS.md
└── benchmarks/
    ├── prediction/  # Task 1
    ├── encoding/    # Task 2
    └── decoding/    # Task 3
```
