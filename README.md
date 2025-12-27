# Titans: Learning to Memorize at Test Time (Titan V3)

> **"Titans: Learning to Memorize at Test Time"** (arXiv:2501.00663) 논문의 공식/연구용 구현체입니다. 
> 본 레포지토리는 최신 **Titan V3** 아키텍처를 적용하여, `torch.func` 기반의 고성능 엔진과 학습 가능한 게이트(Learnable Gates)를 포함하고 있습니다.

## 🌟 주요 특징 (Titan V3)
이 레포지토리는 세 가지 핵심 요소를 통합하여 최상의 성능을 제공합니다:

1.  **Functional Engine (`torch.func`)**: 메모리 업데이트 루프를 순수 함수형으로 구현하여 `vmap`, `grad`를 통해 초고속 병렬 처리가 가능합니다.
2.  **Learnable Gates ($\alpha, \eta, \theta$)**: 데이터에 따라 망각(Forgetting), 모멘텀(Momentum), 학습률(Learning Rate)을 동적으로 조절합니다.
3.  **Unified Architectures**: MAC, MAG, MAL 세 가지 모델 변형 모두 최신 엔진을 사용하도록 최적화되었습니다.

---

## 🚀 설치 및 시작하기

### 설치
```bash
git clone https://github.com/snuconnectome/titans.git
cd titans
pip install -r requirements.txt  # (필요 시)
```

### 기본 사용법 (MAC 모델)
**Memory as Context (MAC)**는 긴 문맥을 처리하는 데 최적화된 모델입니다.

```python
import torch
from titans_pytorch.models.mac import MemoryAsContext

# 모델 초기화
model = MemoryAsContext(
    dim=512, 
    vocab_size=32000, 
    segment_len=128
)

# 입력 데이터 (Batch, SeqLen)
x = torch.randint(0, 32000, (1, 1024))

# 추론 (Forward)
logits = model(x)
print(logits.shape) # torch.Size([1, 1024, 32000])
```

### 아키텍처 선택 가이드
| 모델 | 전체 이름 | 특징 | 추천 용도 |
| :--- | :--- | :--- | :--- |
| **MAC** | Memory as Context | 입력을 세그먼트로 나누어 처리, 긴 문맥 기억에 탁월 | 긴 문서 요약, DNA 서열 분석 |
| **MAG** | Memory as Gate | Attention과 Memory를 병렬로 실행 후 게이팅 | 실시간 스트리밍, 일반 언어 모델링 |
| **MAL** | Memory as Layer | Memory 레이어 후 Attention 레이어 적층 | 복잡한 추론, 심층 모델 |

---

## 🧠 발전 방향: Titan-Neuro (Brain Dynamics Modeling)

우리는 Titans 아키텍처를 단순 언어 모델을 넘어, **4D fMRI 뇌영상 데이터 분석**을 위한 **Titan-Neuro**로 발전시키고 있습니다.

### 🎯 목표
인간의 뇌 활동은 시간적(Temporal)으로 매우 길고 복잡한 역동성을 가집니다. 기존의 Transformer(예: SwiFT)는 긴 시계열 처리에 한계가 있지만, **Titans의 Neural Memory**는 무한에 가까운 시간적 맥락을 $O(1)$ 메모리로 처리할 수 있습니다.

### 🗺 로드맵 (Vision for Titan-Neuro)

#### 1. SwiFT(Swin 4D Transformer)와의 통합
*   **현재**: SwiFT는 4차원 윈도우 Attention을 사용합니다.
*   **미래**: 공간적 특징($x_y, y_t, z_t$)은 Swin Transformer로 추출하고, **시간적 흐름($t$)은 Titans Memory로 모델링**하는 하이브리드 아키텍처를 구축합니다.

#### 2. Grand Budapest Hotel (ds003017) 검증
*   **데이터셋**: [OpenNeuro ds003017](https://openneuro.org/datasets/ds003017/versions/1.0.3) ('그랜드 부다페스트 호텔' 시청 중 fMRI)
*   **활용**: 영화의 복잡한 사회적 상호작용과 시각적 전개에 따른 뇌 반응(Face Processing, Social Cognition)의 시간적 패턴을 학습합니다.
*   **Action**: Titans 모델이 영화의 긴 서사 구조(Narrative Structure)를 기억하고 다음 프레임을 예측할 수 있는지 검증합니다.

### 🧪 실험 준비 (DataLad)
실제 ds003017 데이터를 사용하기 위해서는 `git annex` 또는 `datalad`가 필요합니다.

```bash
# DataLad 설치
sudo apt-get install git-annex
pip install datalad

# 데이터셋 다운로드 (OpenNeuro)
datalad install https://github.com/OpenNeuroDatasets/ds003017.git
cd ds003017
datalad get sub-01/func/sub-01_task-movie_bold.nii.gz
```

---

## 🛠 코드 구조

```
titans/
├── titans_pytorch/
│   ├── memory/
│   │   ├── neural_memory.py  # Titan V3 Core (Learnable Gates)
│   │   └── functional.py     # torch.func Engine
│   ├── models/
│   │   ├── mac.py            # Memory as Context
│   │   ├── mag.py            # Memory as Gate
│   │   └── mal.py            # Memory as Layer
│   └── utils.py
├── tests/                    # pytest 테스트 스위트
└── main.py                   # 검증 스크립트
```

## License
MIT License
