# 📊 Dataset Documentation: Natural Language fMRI (ds003020)

This document provides a comprehensive overview of the `ds003020` dataset, its metadata, and the research tasks possible with the Titan-Neuro architecture.

---

## 1. Dataset Overview

- **Name**: An fMRI dataset during a passive natural language listening task.
- **Accession ID**: [OpenNeuro ds003020](https://openneuro.org/datasets/ds003020)
- **Reference**: LeBel, A., Wagner, L., Jain, S. et al. *A natural language fMRI dataset for voxelwise encoding models*. Sci Data 10, 555 (2023). [doi:10.1038/s41597-023-02437-z](https://doi.org/10.1038/s41597-023-02437-z)
- **Subjects**: 9 healthy adults (sub-UTS01 to sub-UTS09).
- **Task**: Participants listened to ~27 hours of natural stories (The Moth Radio Hour, etc.) and short localizer tasks while inside the scanner.

### 🎥 Stimuli (Input)
- **Audio**: Raw `.wav` files located in the `stimuli/` directory.
- **Transcript**: Aligned transcripts (word/phoneme level) in `derivative/TextGrids/`.
- **Variety**: Over 80 distinct stories (e.g., *A Doll's House*, *Avatar*, *Buck*, *Food*).

### 🧠 fMRI Data (Output)
- **Repetition Time (TR)**: 2.0 seconds.
- **Resolution**: 3T Siemens Skyra, 2.24mm isotropic voxels (typically).
- **Format**: BIDS compliant (`.nii.gz` for raw, `.hf5` for preprocessed).
- **Preprocessed Data**: Located in `derivative/preprocessed_data/`, containing voxel-wise time series aligned to a standard surface or volume.

---

## 2. Metadata & Structure

### BIDS Structure
```text
ds003020/
├── participants.tsv       # Subject age, sex, etc.
├── dataset_description.json
├── sub-UTS01/             # Data for subject UTS01
│   ├── ses-1/             # Multiple sessions (up to 20 per subject)
│   │   └── func/          # fMRI bold files (.nii.gz)
├── stimuli/               # Story audio files (.wav)
└── derivative/            # Processed artifacts
    ├── preprocessed_data/ # Voxel time-series (.hf5)
    ├── TextGrids/         # Aligned word/phoneme labels
    └── respdict.json      # Dictionary of TR counts per story
```

### Key Parameters
- **Repetition Time (TR)**: 2.0s
- **Voxel Count**: ~50,000 to 100,000 (after masking).
- **Alignment**: Preprocessed data is typically provided in a subject-specific "flat" or "standard" space.

---

## 3. Possible Tasks with Titan-Neuro

The **Titans Neural Memory** architecture is uniquely suited for this dataset due to the **long-range temporal dependencies** in natural language.

### ✅ Task 1: Multi-step Brain Trajectory Prediction (Dynamics)
- **Goal**: Given $X_{0:t}$ brain states, predict $X_{t+1}, X_{t+2}, X_{t+3}$.
- **Titans Role**: Use Neural Memory to capture the "state" of the brain as it processes a continuous story.
- **Status**: Logic implemented in `benchmark_leaderboard.py`, needs adaptation for `ds003020`.

### 🧠 Task 2: Brain Encoding (Stimulus → Brain)
- **Goal**: Predict the fMRI response ($B_t$) given the audio features ($S_t$).
- **Features**: Extract embeddings from audio (e.g., Wav2Vec2, Whisper) or text (e.g., GPT, BERT).
- **Titans Role**: Memory acts as a temporal integrator, modeling how linguistic context builds up and affects the Hemodynamic Response (HRF) which has a 4-6s delay.

### 🎧 Task 3: Semantic/Scene Decoding (Brain → Stimulus)
- **Goal**: Identify which story or scene the subject is hearing, or classify the "type" of event (e.g., *Action*, *Dialogue*, *Emotional*).
- **Labels**: Use `_events.tsv` or metadata from `TextGrids`.
- **Titans Role**: Memory integrates noisy fMRI signals over time to produce a stable representation of the current semantic context.

### ✍️ Task 4: Continuous Language Reconstruction
- **Goal**: Reconstruct the actual words or meanings from brain activity (as in Tang et al., 2023).
- **Titans Role**: Use the high-capacity memory to maintain a coherent narrative "thread" during decoding.

---

## 4. Current Implementation Status

| Feature | ds003017 (Budapest) | ds003020 (Natural Lang) |
| :--- | :---: | :---: |
| **Data Loader** | Partially (Mock) | **Planned (NaturalLanguageDataset)** |
| **3D Encoder/Decoder** | Yes | Yes |
| **Neural Memory** | Yes | Yes |
| **Trajectory Bench** | Yes | Planned |
| **Encoding Bench** | Planned | Planned |
| **Decoding Bench** | Planned | Planned |

---
*Created by Titan-Neuro Onboarding Assistant (Dec 30, 2025)*

