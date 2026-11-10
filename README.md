# Motor Imagery EEG Classification Using CSP Features and RBF-SVM

**IEEE DataPort / MathWorks Challenge — SPS Bangalore Chapter, SPS Forum 2026**

A four-class motor-imagery EEG classification project using Common Spatial Pattern (CSP) log-variance features, ANOVA-based feature selection, and a multiclass Support Vector Machine (SVM) with an RBF kernel.

## Team Members

| Name | Institution | Role |
|---|---|---|
| Pranathi CK | MS Ramaiah Institute of Technology | Student |
| Vivikta A | MS Ramaiah Institute of Technology | Student |
| Aisiri P Hegde | MS Ramaiah Institute of Technology | Student |
| Sanjana Nayak | MS Ramaiah Institute of Technology | Student |

## Project Overview

This project investigates the classification of four motor-imagery (MI) states from electroencephalography (EEG) recordings:

- **Left hand**
- **Right hand**
- **Feet**
- **Idle**

The work uses the IEEE DataPort 7-day Motor Imagery BCI dataset. The technical report describes data from 20 subjects, across 7 sessions, with 6 runs per session and 40 trials per run. The implemented pipeline was developed in MATLAB, with Python used for data extraction and preparation.

## Dataset

The dataset described in the project report contains recordings from:

- 20 subjects
- 7 sessions
- 6 runs per session
- 40 trials per run

The labeled training set used for model development contains **33,474 trials**. The challenge test set contains **2,000 trials**.

**Data availability:** The raw EEG dataset is not included in this repository. Users must obtain the data through its original source and comply with the dataset's access, citation, and usage conditions. The repository's software license, if one is added, does not grant rights to the dataset.

## Methodology

The classification workflow consists of the following stages.

### 1. EEG feature construction with CSP

Common Spatial Pattern (CSP) was used to derive spatial features from the EEG signals. CSP log-variance features were calculated in four frequency bands:

- Mu
- Low-beta
- Mid-beta
- High-beta

Features were computed for every pairwise combination of the four classes, retaining four CSP components for each pairwise comparison. This produced **96 candidate features per trial**:

`4 frequency bands × 6 class pairs × 4 components = 96 features`

### 2. Feature selection using ANOVA

A one-way ANOVA F-statistic was calculated for each candidate feature across the four classes. The **72 highest-ranked features** were retained for the final model. The report states that this feature count was selected empirically using earlier cross-validation.

### 3. Feature standardization

Features were z-score standardized using the training-set mean and standard deviation. The same training statistics were applied during inference; test data were not independently re-standardized.

### 4. Classification using RBF-SVM

A multiclass Support Vector Machine was trained using MATLAB's `fitcecoc` with one-vs-one error-correcting output codes. Each binary learner used an RBF kernel, automatic kernel scaling, and a box constraint of 1.

### 5. Cross-session and inter-subject handling

Features were pooled and ranked across subjects and sessions rather than calibrated separately for each subject. Global training-set standardization was used so inference would not require new session-specific standardization statistics.

The report notes that no explicit subject- or session-specific adaptation layer was implemented.

## Results and Performance

| Metric | Reported result |
|---|---:|
| Labeled training trials | 33,474 |
| Test-set trials | 2,000 |
| Candidate CSP features | 96 |
| Selected features | 72 |
| Five-fold stratified cross-validation accuracy | **41.69% ± 0.82%** |
| Four-class chance baseline (balanced classes) | 25% |

The reported accuracy is from **five-fold stratified cross-validation** on the labeled training data. Feature ranking and standardization were refitted within each fold to avoid information leakage. It is not a measured accuracy on the 2,000-trial test set.

The test-set prediction counts reported in the technical report were:

| Class | Predictions |
|---|---:|
| Left hand | 509 |
| Right hand | 503 |
| Feet | 499 |
| Idle | 489 |
| **Total** | **2,000** |

The report identifies low-beta features as prominent among the most discriminative features. It also notes that strong individual feature F-statistics did not translate into high overall four-class classification accuracy.

## Edge-Inference Feasibility (Simulation)

No physical embedded device was available for testing. Instead, the MATLAB prediction function (`predictBCI.m`) was compiled to native C using MATLAB Coder with a MEX target and timed on host hardware.

The report records:

- Host compiled inference: **13.15 ms per trial** (approximately 76.0 trials/second).
- Raspberry Pi 4-class estimate: **52.6 ms per trial** (approximately 19.0 trials/second), based on an estimated 4× scaling factor.

The Raspberry Pi figure is an estimate, not a measurement on a physical Raspberry Pi. The report identifies model size (approximately 60 MB, attributed to the RBF-SVM support vectors) as a more significant deployment constraint than inference speed.

## Technologies Used

- **MATLAB** — CSP/SVM model workflow and prediction.
- **MATLAB Statistics and Machine Learning Toolbox** — `fitcecoc` multiclass SVM.
- **MATLAB Coder** — MEX compilation for host inference timing.
- **Python** — data extraction and preparation.
- **NumPy** — array handling and `.npz` loading.
- **SciPy** — signal processing and MATLAB-compatible file export.
- **pandas** — data handling.

The project report also lists Python standard-library modules such as `os`, `glob`, `re`, and `collections` for file traversal and organization.

## Repository Structure

The suggested structure below is a guide. Adjust folder names to match the actual files committed to this repository.

```text
motor-imagery-eeg-classification/
├── README.md
├── matlab/
│   ├── feature_extraction/
│   ├── training/
│   ├── prediction/
│   └── evaluation/
├── python/
│   └── data_preparation/
├── report/
│   ├── Technical_Report.pdf
│   └── README.pdf
├── results/
│   └── figures/
├── data/
│   └── README.md
└── .gitignore
```

Raw EEG recordings and other restricted or very large source data should not be committed. Add only files that the team is permitted to redistribute.

## Running the Project

The technical report describes the processing and model configuration, but it does not provide a complete, verified set of command-line instructions or exact script paths. Therefore, the following is a high-level workflow, not a guaranteed one-command setup:

1. Obtain the dataset from its original source and place it in the location expected by the data-preparation scripts.
2. Run the Python data-preparation scripts, if included, to prepare the EEG data for the MATLAB workflow.
3. Open MATLAB and run the feature extraction and training scripts included in the repository.
4. Run the prediction script using the saved model and the same training-set standardization statistics.
5. Use the evaluation scripts and supplied results to review model performance.

Before publishing, add the exact script names, required MATLAB toolboxes/versions, expected input formats, and execution order after checking the final code files.

## Challenges and Limitations

The technical report identifies the following limitations:

- Overall cross-validation accuracy is moderate (41.69% ± 0.82%), despite being above the 25% balanced four-class chance baseline.
- No explicit per-subject or per-session adaptation layer was implemented.
- High individual feature F-statistics did not guarantee strong multiclass performance.
- The final model was refit using all labeled training trials; the reported 41.69% result comes from a separate cross-validation procedure, not evaluation of those exact final deployed weights.
- The RBF-SVM model is relatively large (approximately 60 MB according to the report).
- Embedded-device timing was estimated from host measurements; no physical Raspberry Pi test was performed.

## Future Work

Potential next steps identified in the report include:

- Subject-specific calibration.
- Deep-learning-based spatial-temporal feature extraction, such as EEGNet-style architectures.
- Trial-level ensembling across sessions.

## References and Acknowledgements

- IEEE DataPort 7-day Motor Imagery BCI dataset — consult the dataset's original page for its citation and usage requirements.
- IEEE DataPort / MathWorks Challenge, SPS Bangalore Chapter, SPS Forum 2026.
- The accompanying project technical report documents the methodology, evaluation, challenges, and simulation-based edge-inference analysis.
