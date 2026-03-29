# Neural Network Multipath Fading Equalization

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Analysis](#2-dataset-analysis)
3. [Pre-processing](#3-pre-processing)
4. [Architecture Justification](#4-architecture-justification)
5. [Experiments](#5-experiments)
6. [Results and Analysis](#6-results-and-analysis)
7. [Plots](#7-plots)
8. [Dataset Split](#8-dataset-split)
9. [Project Structure](#9-project-structure)

---

## 1. Problem Statement

In a wireless communication system, the transmitted signal passes through a multipath fading channel and is further corrupted by additive white Gaussian noise (AWGN). The received signal model is:

```
r[n] = h[n] * s[n] + w[n]
```

| Symbol | Meaning |
|---|---|
| `s[n]` | Transmitted QAM symbol |
| `h[n]` | Multipath fading channel (Rayleigh or Rician) |
| `w[n]` | Additive white Gaussian noise |
| `r[n]` | Received (corrupted) signal |

The receiver must equalize `r[n]` to recover `s[n]`. Traditional equalizers such as MMSE and Zero-Forcing use fixed mathematical formulas derived from channel statistics. This project replaces them with a data-driven neural network that learns the equalization function directly from examples, conditioned on the modulation scheme, channel type, and SNR.

---

## 2. Dataset Analysis

**Source:** [5G/6G Multipath Fading Equalization Dataset — Kaggle](https://www.kaggle.com/datasets/programmer3/5g6g-multipath-fading-equalization-dataset)

### Basic Statistics

| Property | Value |
|---|---|
| Total samples | 6,573 |
| Signal length per sample | 100 complex numbers |
| Modulation schemes | QPSK (2,159) · QAM-16 (2,224) · QAM-64 (2,190) |
| Channel models | Rayleigh (3,280) · Rician (3,293) |
| SNR range | −4.99 dB to +29.99 dB |

### Column Descriptions

| Column | Type | Role | Description |
|---|---|---|---|
| `Modulation_Type` | String | Input | QPSK / QAM-16 / QAM-64 |
| `Channel_Model` | String | Input | Rayleigh / Rician |
| `SNR_dB` | Float | Input | Signal-to-Noise Ratio in dB |
| `Received_Signal` | String | Input | 100 complex numbers — corrupted signal |
| `Denoised_Signal` | String | Target | 100 complex numbers — ground-truth equalized signal |
| `BER_Before` | Float | Analysis only | Bit Error Rate before equalization |
| `BER_After` | Float | Analysis only | Bit Error Rate after equalization |

> `BER_Before` and `BER_After` are not used as model inputs. They are used only in the exploratory analysis section to confirm that equalization is effective in the dataset.

### Key Observation — Residual to Signal Ratio

```
Mean |denoised − received|  =  0.085  units
Mean |received|             =  59.7   units
Ratio                        =  0.0014
```

The correction the equalizer applies is only 0.14% of the signal magnitude. This observation directly shapes the modelling strategy described in Section 3.

### Class Balance

The dataset is well balanced across all three modulation schemes and both channel types. No class weighting or oversampling is required.

---

## 3. Pre-processing

### 3.1 Signal String Parsing

The `Received_Signal` and `Denoised_Signal` columns store numpy complex128 arrays as printed strings. A custom parser handles four token formats that appear in the data:

| Case | Example | Interpretation |
|---|---|---|
| Fused complex | `-31.61-13.65j` | real = −31.61, imag = −13.65 |
| Standalone imaginary | `-4.76j` | imaginary part of preceding real token |
| Separated pair | `-40.37` then `-4.76j` | standard numpy print format |
| Pure real | `-40.37` | imaginary part arrives in next token |

All 6,573 rows parse to exactly 100 complex numbers each, verified with assertions before any training.

### 3.2 Residual Learning

Rather than predicting the full denoised signal, the model is trained to predict the **residual**:

```
target = denoised_signal − received_signal
```

The residual has a standard deviation of approximately 0.07 units, while the received signal has a standard deviation of approximately 57 units — a factor of ~800× difference. Predicting a near-zero correction is far easier than reconstructing the full 60-unit signal from scratch. The final prediction is recovered as:

```
equalized_output = received_signal + predicted_residual
```

This mirrors the principle behind residual networks — the model only learns the delta, not the full mapping.

### 3.3 Categorical Encoding

```
Modulation_Type : QAM-16 → 0,  QAM-64 → 1,  QPSK → 2
Channel_Model   : Rayleigh → 0,  Rician → 1
```

### 3.4 Feature Matrix Construction

Each complex signal is split into real and imaginary components:

```
X (input)   shape (N, 203)
             ├── meta     (3)  :  [mod_encoded, channel_encoded, snr_db]
             └── signal (200)  :  [rx_real × 100 | rx_imag × 100]

y (target)  shape (N, 200)
             [residual_real × 100 | residual_imag × 100]
```

### 3.5 Scaling

Three separate `StandardScaler` instances are used because the signal and its residual have vastly different magnitudes:

| Scaler | Applied to | Approximate std |
|---|---|---|
| `meta_scaler` | `[mod_code, ch_code, snr_db]` | ~0.8 |
| `sig_scaler` | Received signal columns | ~57 |
| `res_scaler` | Residual target columns | ~0.07 |

Using a single scaler for signal and residual together would make the residual near-zero in normalised space, causing the network to learn almost nothing. All three scalers are fitted on the full dataset before splitting, then applied identically to all splits. This simulates known channel statistics being available at the receiver, which is standard practice in communication systems.

---

## 4. Architecture Justification

Three architectures were designed and compared. Each builds on the limitations identified in the previous one.

### 4.1 MLP — Baseline

```
Input (203) → FC(256) → BN → GELU → Dropout(0.35)
            → FC(256) → BN → GELU → Dropout(0.35)
            → FC(256) → BN → GELU → Dropout(0.35)
            → FC(128) → BN → GELU → Dropout(0.35)
            → FC(128) → BN → GELU → Dropout(0.35)
            → Output (200)

Trainable parameters: 261,064
```

Included as a baseline. Treats all 203 input features uniformly with no structural inductive bias. Expected to establish the performance floor.

### 4.2 ResNet — Residual Network

```
Input (203) → Linear(256) → BN → GELU
            → [ResBlock × 4]
            → BN → GELU → Output (200)

ResBlock:
    input → BN → GELU → Linear(256) → BN → GELU → Dropout → Linear(256) → + input

Trainable parameters: 635,080
```

**Why chosen:**

Pre-activation residual blocks are particularly suited to this task for two reasons.

First, the skip connection `output = input + learned_correction` structurally aligns with the residual learning objective. If a block learns nothing useful, the identity path passes the signal through unchanged — the architecture and the task formulation reinforce each other.

Second, pre-activation order (normalisation and activation before the linear transformation, not after) provides more stable gradients through depth. This matters because the residual targets are very small in absolute value, making gradients naturally small and prone to vanishing without careful architecture choices.

GELU activation is used throughout rather than ReLU because GELU is smooth near zero with a small non-zero gradient for negative inputs. Since residual targets are near zero, this smoothness aids convergence.

### 4.3 FiLM-ResNet — Feature-wise Linear Modulation

```
Signal path: rx (200) → Linear(256) → BN → GELU
             → [FiLMResBlock × 4] → BN → GELU → Output (200)

Meta path:   [mod, ch, snr] fed into every FiLMResBlock

FiLMResBlock:
    h     → BN → GELU → Linear → FiLM(meta) → BN → GELU → Dropout → Linear → + h
    meta  → Linear(64) → GELU → Linear(512) → split → [gamma | beta]
    FiLM  → h_out = gamma × h + beta

Trainable parameters: 768,456
```

**Why this is the most principled architecture:**

The equalization problem is fundamentally condition-dependent. The optimal filter for QPSK (4 constellation points, wide spacing) is very different from QAM-64 (64 constellation points, narrow spacing). Rayleigh fading (purely scatter-based, no line-of-sight) behaves differently from Rician fading (dominant line-of-sight component). And equalization at SNR = −5 dB requires aggressive correction while at SNR = +30 dB almost nothing needs to change.

A plain MLP or ResNet must find one set of weights that averages across all of these conditions. FiLM instead learns condition-specific modulation parameters that adjust the hidden representation inside every residual block:

```
h_conditioned = γ(modulation, channel, SNR) × h_signal + β(modulation, channel, SNR)
```

The `γ` and `β` values are different for every sample, computed from that sample's meta-features. This lets the network effectively implement different equalization strategies for different conditions while sharing the bulk of its parameters.

FiLM weights are initialised so that `γ = 1` and `β = 0` at epoch 0 (identity transform). The network departs from identity only where the meta-features provide useful signal, making training stable from the beginning.

### 4.4 Training Configuration

| Setting | Value | Justification |
|---|---|---|
| Loss | `HuberLoss(delta=0.5)` | Quadratic for small errors like MSE, linear for large outliers. More robust than pure MSE on small residuals |
| Optimiser | `AdamW(lr=1e-3, wd=5e-4)` | AdamW decouples weight decay from gradient scale, giving better regularisation than plain Adam |
| LR schedule | `CosineAnnealingLR` (LR → LR/50) | Smooth decay; avoids abrupt plateaus near convergence |
| Early stopping | Patience = 30 | Stops when validation loss stops improving; best checkpoint restored automatically |
| Gradient clipping | `max_norm = 1.0` | Prevents rare large gradient updates from destabilising the small-residual training signal |
| Batch size | 256 | Larger batches provide more stable BatchNorm statistics |
| Dropout | 0.35 | Stronger regularisation to reduce the train/val gap observed with the smaller dataset |

---

## 5. Experiments

Three experiments are run sequentially. All use the **same held-out test set**, so all RMSE, R², and NMSE numbers in the results table are directly comparable.

### Part A — Baseline

Trains MLP, ResNet, and FiLM-ResNet on the original 6,573 samples. Establishes the performance baseline.

### Part B — Phase-Rotation Augmentation

Phase rotation is a physically valid augmentation for complex wireless signals. Multiplying a complex signal by `e^(jθ)` is equivalent to a different carrier phase offset at the transmitter — the noise statistics, channel type, and equalization residual all remain valid under the same rotation.

Three rotation angles are applied to the training split only, growing it from 4,732 to 18,928 samples (4×). The best model from Part A is then retrained on this augmented set.

**Why validation and test sets are not augmented:**

The validation set is used to guide early stopping. If it is augmented, early stopping responds to a different signal than it would in production, causing the saved checkpoint to be tuned to inflated data rather than real-world generalisation.

The test set is the final authority on model performance. If it is augmented with rotations of training data, the model may have seen structurally identical samples during training, artificially inflating the reported metrics. The comparison between Part A and Part B would then be meaningless — Part B would appear better simply because its test set is easier, not because it generalises better.

Keeping validation and test sets unchanged means the NMSE improvement from Part A to Part B is a real, honest measurement of generalisation improvement from augmentation.

### Part C — Specialist Models

One ResNet is trained per modulation scheme (QPSK, QAM-16, QAM-64) using only that modulation's data. Each specialist is tested on its own modulation's held-out samples. A combined overall metric is computed by concatenating all three specialists' test predictions, enabling fair comparison with the joint models.

---

## 6. Results and Analysis

### 6.1 Part A — Baseline

| Model | MSE | RMSE | MAE | R² | NMSE (dB) |
|---|---|---|---|---|---|
| MLP | 0.004009 | 0.063314 | 0.050874 | 0.084 | −0.41 dB |
| ResNet | **0.001760** | **0.041954** | **0.032863** | **0.602** | **−3.98 dB** |
| FiLM-ResNet | 0.001761 | 0.041969 | 0.032958 | 0.603 | −3.98 dB |

**Best model: ResNet** (RMSE = 0.04195, NMSE = −3.984 dB)

The MLP performs poorly — R² of 0.08 means it explains only 8% of the variance in the residuals. Its training and validation losses tracked closely throughout training, indicating the model was underfitting rather than overfitting. Treating all 203 inputs uniformly with no structural bias is insufficient for this task.

ResNet and FiLM-ResNet perform nearly identically despite FiLM's more sophisticated conditioning architecture. The most likely cause is dataset scale — with roughly 110 samples per (modulation, channel, SNR) condition, the FiLM sub-network does not have enough examples of each condition to learn meaningful `(γ, β)` adjustments and defaults to near-identity transforms.

### 6.2 Part B — Augmentation Results

| Metric | Part A — ResNet | Part B — ResNet (aug) | Change |
|---|---|---|---|
| MSE | 0.001760 | 0.001457 | ✓ improved |
| RMSE | 0.041954 | 0.038170 | ✓ improved |
| MAE | 0.032863 | 0.029957 | ✓ improved |
| R² | 0.602 | 0.664 | ✓ improved |
| NMSE | −3.984 dB | **−4.805 dB** | ✓ improved by 0.82 dB |

Augmentation improved every metric. NMSE improved by 0.82 dB and R² rose from 0.60 to 0.66. The training set grew from 4,732 to 18,928 samples, giving the ResNet more diverse signal patterns to learn from. The gain is genuine — it is measured on the same clean test set as Part A.

### 6.3 Part C — Specialist Results

| Model | N Test | RMSE | R² | NMSE (dB) |
|---|---|---|---|---|
| Specialist QPSK | 324 | 0.04816 | 0.469 | −2.789 dB |
| Specialist QAM-16 | 334 | 0.04680 | 0.500 | −3.038 dB |
| Specialist QAM-64 | 329 | 0.04760 | 0.485 | −2.888 dB |
| Combined overall | 987 | 0.04752 | 0.487 | −2.905 dB |

The specialist models underperformed the joint ResNet (−2.91 dB vs −3.98 dB). Each specialist receives only ~1,500 training samples after splitting by modulation — too few for a 635,080-parameter network to learn well. The joint model benefits from shared representations: the fading channel mechanism is the same regardless of modulation, so sharing weights across modulations is beneficial at this data scale.

### 6.4 Final Comparison — All Approaches

| Approach | RMSE | R² | NMSE (dB) |
|---|---|---|---|
| Part A — MLP | 0.06331 | 0.084 | −0.41 dB |
| Part A — ResNet | 0.04195 | 0.602 | −3.98 dB |
| Part A — FiLM-ResNet | 0.04197 | 0.603 | −3.98 dB |
| **Part B — ResNet (augmented)** | **0.03817** | **0.664** | **−4.81 dB** |
| Part C — Specialist QPSK | 0.04816 | 0.469 | −2.79 dB |
| Part C — Specialist QAM-16 | 0.04680 | 0.500 | −3.04 dB |
| Part C — Specialist QAM-64 | 0.04760 | 0.485 | −2.89 dB |
| Part C — Specialists (combined) | 0.04752 | 0.487 | −2.91 dB |

**Winner: Part B — ResNet with phase-rotation augmentation.**

### 6.5 Metric Interpretation Guide

| Metric | What it measures |
|---|---|
| **MSE** | Mean squared error between predicted and true residuals (original signal scale) |
| **RMSE** | Square root of MSE — in the same units as the signal |
| **MAE** | Mean absolute error — less sensitive to outliers than RMSE |
| **R²** | Fraction of residual variance explained by the model (0 = baseline, 1 = perfect) |
| **NMSE (dB)** | Standard wireless comms metric: `10 log₁₀(‖pred − true‖² / ‖true‖²)` |

NMSE scale reference:

| NMSE | Meaning |
|---|---|
| 0 dB | Error equals signal power — model adds no value |
| −3 dB | Error is half signal power — poor |
| −10 dB | Error is 10% of signal power — good |
| −20 dB | Error is 1% of signal power — excellent |

The best result of −4.81 dB means the model's prediction error carries about 33% of the signal power. It is learning real equalization patterns and consistently outperforms no-equalization, but would need more training data to reach production-grade performance.

---

## 7. Plots

The notebook generates the following plots automatically during training:

| Filename | Description |
|---|---|
| `eda.png` | 6-panel EDA — modulation distribution, channel distribution, SNR histogram, BER vs SNR scatter, IQ constellation comparison, residual magnitude histogram |
| `partA_training_curves.png` | Train and validation Huber loss per epoch for MLP, ResNet, and FiLM-ResNet |
| `partA_equalizer_output.png` | Predicted vs true scatter plot, IQ constellation comparison, symbol-level time series |
| `partA_nmse.png` | NMSE vs SNR for all three models, best model broken down per modulation, best model broken down per channel type |
| `partB_training_curves.png` | Part A vs Part B training curves overlaid — shows the effect of augmentation on convergence speed and final loss |
| `partC_training_curves.png` | Training curves for each of the three specialist models |
| `partC_comparison.png` | Left panel: NMSE vs SNR for specialists vs joint model. Right panel: bar chart of NMSE across all approaches |

---

## 8. Dataset Split

The split satisfies the assignment requirement of **80:20 train:test**.

```python
# Step 1 — reserve 20% as the test set
idx_tr_tmp, idx_te = train_test_split(idx_all, test_size=0.20, random_state=42)

# Step 2 — split remaining 80% into train and validation
idx_tr, idx_val    = train_test_split(idx_tr_tmp, test_size=0.10, random_state=42)
```

| Split | Proportion of total | Samples |
|---|---|---|
| Train | 72% | 4,732 |
| Validation | 8% | 526 |
| **Test** | **20%** | **1,315** |

Validation is carved from the training portion so the test boundary remains a clean 20%. The 526-sample validation set is sufficient for early stopping — it only needs to provide a stable signal of whether generalisation is improving, not a large-scale evaluation.

For Part B, augmentation grows only the training split:

```
Train:  4,732 → 18,928  (4× via phase rotation at π/6, π/3, π/2)
Val  :    526  unchanged
Test : 1,315  unchanged — same samples as Part A
```

---

## 9. Project Structure

```
.
├── Neural Network Multipath Fading Equalization.ipynb          # Main notebook (Parts A, B, C)
├── README.md                      # This file
│
└── output/                       # Generated on first run
    ├── eda.png
    ├── partA_training_curves.png
    ├── partA_equalizer_output.png
    ├── partA_nmse.png
    ├── partB_training_curves.png
    ├── partC_training_curves.png
    ├── partC_comparison.png

---
