# Feature Extraction and Neural Network Architecture for Deep Audio Zooming

This document describes the feature extraction pipeline and neural network architecture used in the deep audio zooming system - a beamwidth‑controllable multichannel speech enhancement model.

---

## Table of Contents

- [Overview](#overview)
- [Feature Extraction and Preprocessing](#feature-extraction-and-preprocessing)
- [Neural Network Architecture Diagram](#neural-network-architecture-diagram)
- [Layer-by-Layer Description](#layer-by-layer-description)
- [Training and Optimization](#training-and-optimization)
- [Model Outputs](#model-outputs)
- [Key Design Choices](#key-design-choices)
- [Computational Efficiency](#computational-efficiency)

---

## Overview

The deep audio zooming system takes multi-channel audio (8 microphones) and a user-defined field of view (FOV) as input, and outputs an enhanced signal containing only sounds from within that FOV. The system combines physics-based feature extraction with a neural beamformer to achieve real-time performance with low computational cost.

---

## Feature Extraction and Preprocessing

### Input: Raw Multi-channel Audio
- 8-channel microphone signals at 16 kHz sampling rate
- Each channel processed independently through Short-Time Fourier Transform (STFT)

### Step 1: Short-Time Fourier Transform (STFT)
- **Window:** Hann window (512-point FFT)
- **Hop length:** 256 samples (50% overlap)
- **Output:** Complex spectrograms Y_c(t,f) for each channel c = 1,...,8
- **Dimensions:** (batch, 8, freq_bins=257, time_frames)

### Step 2: Log Power Spectrum (LPS)
- Taken from reference microphone (channel 1)
- L(t,f) = log(|Y₁(t,f)|² + ε)
- **Purpose:** Provides spectral information about the sound (harmonics, formants, energy distribution)
- **Output shape:** (batch, freq, time)

### Step 3: Inter-channel Phase Difference (IPD)
- Computed between reference microphone (channel 1) and each other microphone c = 2,...,8
- IPD_c(t,f) = (Y_c/|Y_c|) · conj(Y₁/|Y₁|)
- **Properties:** Complex-valued, unit magnitude, encodes phase difference
- **Purpose:** Contains spatial/directional information about sound sources
- **Output shape:** (batch, 7, freq, time) complex

### Step 4: Theoretical IPD for Candidate Directions
- For each candidate direction θ (azimuth angle, e.g., sampled every 1° from -180° to 180°):
  - Compute time difference of arrival (TDOA) based on microphone geometry
  - τ_c(θ) = (direction vector · (mic_position_c - mic_position_ref)) / speed_of_sound
  - Theoretical IPD: Γ_θ^{c,theory}(f) = exp(-j·2π·f·τ_c(θ))
- **Purpose:** Provides reference for comparing observed IPD

### Step 5: Directional Similarity Score
- For each candidate direction θ:
  - d_θ(t,f) = Re( Σ_{c=2}^8 IPD_c(t,f) · conj(Γ_θ^{c,theory}(f)) )
- **Range:** -1 to +1 (higher means better alignment)
- **Interpretation:** High values indicate T-F bin dominated by source from direction θ
- **Output shape:** (batch, num_candidates, freq, time)

### Step 6: FOV Pooling
- Given user-defined FOV azimuth range [Θ_l, Θ_h]:
  - D_in(t,f) = max over θ inside FOV of d_θ(t,f)
  - D_out(t,f) = max over θ outside FOV of d_θ(t,f)
- **Purpose:** Condenses directional information into two scalars per T-F bin
- **Output shape:** (batch, freq, time) for each

### Step 7: Final Input Feature Vector
- For every time-frequency bin, concatenate:
  - L(t,f): Log Power Spectrum (1 value)
  - D_in(t,f): Inside-FOV feature (1 value)
  - D_out(t,f): Outside-FOV feature (1 value)
- **Final input shape:** (batch, freq, time, 3)
- **This 3-dimensional vector is fed directly to the neural network**

---

## Neural Network Architecture Diagram


---

## Layer-by-Layer Description

### Layer 1: GRU-1 (Temporal Processing)

| Property | Value |
|----------|-------|
| **Type** | Unidirectional Gated Recurrent Unit (GRU) |
| **Input** | 3-dim features [LPS, D_in, D_out] over time |
| **Input shape** | (batch * freq, time, 3) |
| **Hidden size** | 128 |
| **Output** | Hidden state h₁(t,f) |
| **Output shape** | (batch, freq, time, 128) |
| **Processing** | Each frequency band processed independently |
| **Purpose** | Captures temporal speech dynamics (phoneme durations, speech patterns) |

**Why GRU?**
- Fewer parameters than LSTM (no output gate)
- Faster computation
- Comparable performance for speech tasks
- Suitable for real-time applications

---

### Layer 2: Mask Estimation

| Property | Value |
|----------|-------|
| **Type** | Two independent linear layers |
| **Input** | GRU-1 hidden state (128-dim) |
| **Input shape** | (batch * freq * time, 128) |
| **Output** | Complex masks M_in(t,f) and M_out(t,f) |
| **Output shape** | (batch, freq, time) complex each |
| **Output representation** | Real and imaginary parts (2 values per mask) |

**Mask Functions:**
- **M_in:** Enhances time-frequency bins dominated by sources inside the FOV
- **M_out:** Represents interference from sources outside the FOV

**Why complex masks?**
- Phase information is crucial for beamforming
- Complex masks can modify both magnitude and phase
- Enables more precise separation than real-valued masks

---

### Layer 3: Apply Masks to All Microphones

| Property | Value |
|----------|-------|
| **Operation** | S_in,c = M_in × Y_c (for each microphone c) |
| **Operation** | S_out,c = M_out × Y_c (for each microphone c) |
| **Input masks** | M_in, M_out: (batch, freq, time) complex |
| **Input microphones** | Y_c: (batch, 8, freq, time) complex |
| **Output** | S_in_all, S_out_all: (batch, 8, freq, time) complex each |
| **Purpose** | Generate initial estimates of target and interference signals |

**Why apply to all microphones?**
- Preserves full spatial information
- Allows network to learn how masks affect each channel
- Essential for subsequent beamforming

---

### Layer 4: Subband Embedding

| Property | Value |
|----------|-------|
| **Input** | S_in_all and S_out_all (complex, all microphones) |
| **Operation** | Extract real and imaginary parts |
| **Components** | • S_in from 8 mics: 8 × 2 = 16 values<br>• S_out from 8 mics: 8 × 2 = 16 values |
| **Concatenation** | Join all 32 values |
| **Output shape** | (batch, freq, time, 32) |
| **Purpose** | Create rich representation for refinement stage |

**Embedding contents per T-F bin:**

**Note:** Original microphone signals Y_c are NOT included - only the masked versions.

---

### Layer 5: Layer Normalization

| Property | Value |
|----------|-------|
| **Type** | Layer Normalization |
| **Input** | 32-dim embedding |
| **Input shape** | (batch, freq, time, 32) |
| **Operation** | Normalize across the 32 feature dimensions |
| **Output shape** | (batch, freq, time, 32) |
| **Purpose** | Stabilizes training, reduces internal covariate shift |

**Why LayerNorm?**
- Works well with recurrent networks
- Reduces sensitivity to initialization
- Speeds up convergence

---

### Layer 6: Linear Projection

| Property | Value |
|----------|-------|
| **Type** | Fully connected linear layer |
| **Input** | 32-dim normalized embedding |
| **Output** | 128-dim representation |
| **Output shape** | (batch, freq, time, 128) |
| **Purpose** | Projects to hidden size for GRU-2 |

**Why project to 128?**
- Matches GRU-2 hidden size
- Provides sufficient capacity for refinement
- Consistent dimensionality throughout network

---

### Layer 7: Leaky ReLU Activation

| Property | Value |
|----------|-------|
| **Type** | Leaky Rectified Linear Unit |
| **Negative slope** | 0.01 |
| **Input** | 128-dim projection |
| **Output** | Same shape, non-linearly transformed |
| **Purpose** | Introduces non-linearity while preserving negative values |

**Why Leaky ReLU instead of ReLU?**
- Audio features can have negative values
- Phase information is important and can be negative
- Prevents "dying neurons" (neurons that always output zero)
- Small negative slope (0.01) preserves information

---

### Layer 8: GRU-2 (Per-Frequency RNN)

| Property | Value |
|----------|-------|
| **Type** | GRU cell applied per frequency band |
| **Input** | 128-dim over time (after projection + Leaky ReLU) |
| **Input shape** | (freq, batch, time, 128) |
| **Hidden size** | 128 |
| **Processing** | Each frequency band independently using GRU cell |
| **Output** | Refined 128-dim representation |
| **Output shape** | (batch, freq, time, 128) |
| **Purpose** | Frequency-specific refinement of representations |

**Why a second GRU?**
- First GRU processes raw features
- Second GRU processes enriched embedding with mask information
- Creates hierarchical representation
- Allows refinement after initial separation

**Why per-frequency processing?**
- Different frequencies have different acoustic characteristics
- Speech harmonics vary across frequency
- More efficient than full 2D processing

---

### Layer 9: Filter Estimation

| Property | Value |
|----------|-------|
| **Type** | Linear layer |
| **Input** | GRU-2 output (128-dim) |
| **Input shape** | (batch * freq * time, 128) |
| **Output** | Complex beamforming weights W_c(t,f) |
| **Output shape** | (batch, freq, time, 8) complex |
| **Output representation** | One complex weight per microphone per T-F bin |

**What the weights represent:**
- Each weight W_c(t,f) determines how much microphone c contributes at T-F bin (t,f)
- Complex nature allows both gain and phase adjustment
- Learned end-to-end through optimization

---

### Layer 10: Beamforming

| Property | Value |
|----------|-------|
| **Operation** | Ŝ(t,f) = Σ_{c=1}^8 W_c*(t,f) · Y_c(t,f) |
| **Input filters** | W_c: (batch, freq, time, 8) complex |
| **Input signals** | Y_c: (batch, 8, freq, time) complex |
| **Output** | Enhanced spectrogram Ŝ(t,f) |
| **Output shape** | (batch, freq, time) complex |
| **Purpose** | Adaptive combination of all microphone channels |

**Why this beamforming operation?**
- Standard MVDR (Minimum Variance Distortionless Response)-like formulation
- Conjugate ensures proper phase alignment
- Weighted sum achieves spatial filtering
- Learnable weights can outperform traditional beamformers

---

### Layer 11: Inverse STFT (Output Layer)

| Property | Value |
|----------|-------|
| **Type** | Inverse Short-Time Fourier Transform |
| **Input** | Enhanced spectrogram Ŝ(t,f) complex |
| **Window** | Hann window (same as analysis) |
| **Hop length** | 256 samples |
| **Output** | Time-domain waveform ŝ(n) |
| **Output shape** | (batch, samples) |
| **Purpose** | Convert enhanced spectrogram back to audible waveform |

---

## Training and Optimization

### Loss Function

The model is trained end-to-end with a combination of two losses:

#### 1. Negative SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
- **Domain:** Time domain
- **Formula:** SI-SDR = 20 log₁₀(||α·s_target|| / ||ŝ - α·s_target||)
- **α:** Optimal scaling factor = (ŝᵀ·s_target) / ||s_target||²
- **Loss:** -SI-SDR (minimize negative to maximize SI-SDR)
- **Purpose:** Measures signal quality independent of scale

#### 2. Spectral L1 Loss
- **Domain:** Frequency domain (magnitude spectra)
- **Formula:** || |Ŝ(t,f)| - |S_target(t,f)| ||₁
- **Purpose:** Ensures spectral accuracy

#### Total Loss
- **L_total = L_SI-SDR + λ · L_spectral**
- **λ:** Weighting factor, typically 0.1

### Training Strategy: Random Field Sampling

| Aspect | Description |
|--------|-------------|
| **What** | Randomly sample a new FOV for each utterance during training |
| **FOV width** | Uniformly between 30° and 180° |
| **FOV centre** | Uniformly random between -180° and 180° |
| **Feature update** | Recompute D_in and D_out based on new FOV |
| **Target update** | Target = anechoic sum of speakers inside new FOV |
| **Purpose** | Forces model to learn beamwidth controllability |

### Optimization Details

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Learning rate** | 0.001 |
| **Batch size** | 8-16 |
| **Gradient clipping** | Max norm 5.0 |
| **Epochs** | ~100 |
| **Hardware** | Single NVIDIA GPU |

---

## Model Outputs

| Output | Shape | Description |
|--------|-------|-------------|
| **Enhanced waveform** | (batch, samples) | Final time-domain output signal for listening |
| **Enhanced spectrogram** | (batch, freq, time) complex | Frequency-domain representation (used for loss) |
| **Mask M_in** | (batch, freq, time) complex | Inside-FOV separation mask |
| **Mask M_out** | (batch, freq, time) complex | Outside-FOV separation mask |
| **Filters W_c** | (batch, freq, time, 8) complex | Beamforming weights per microphone |

---

## Key Design Choices

| Choice | Reason |
|--------|--------|
| **Subband processing** | Reduces complexity, enables frequency-specific learning |
| **GRU over LSTM** | Fewer parameters, faster computation, comparable performance |
| **Two GRU stages** | Hierarchical processing: first capture patterns, then refine |
| **Complex masks/filters** | Phase information is critical for beamforming |
| **Layer Normalization** | Stabilizes training, works well with RNNs |
| **Leaky ReLU** | Preserves negative values (important for phase information) |
| **FOV pooling** | Condenses directional info, enables beamwidth control |
| **Random field sampling** | Teaches adaptability to different beamwidths |
| **Apply masks to all mics** | Preserves full spatial information |
| **S_in/S_out only in embedding** | Focus on separated signals, more compact |

---

## Computational Efficiency

| Metric | Value |
|--------|-------|
| **Parameters** | ~860,000 |
| **MACs/second** | 184 MMACs/s (Million Multiply-Accumulates per second) |
| **Latency** | ~10-20 ms |
| **Memory footprint** | ~5-10 MB |

### Suitable for Real-time Deployment On:
- Hearing aids
- Smart speakers
- Mobile devices
- Conference systems
- Wearable audio devices
- Embedded systems

---

## Summary

The deep audio zooming system combines:
1. **Physics-based feature extraction** (LPS, IPD, directional features, FOV pooling)
2. **Neural network processing** (two GRU stages, mask estimation, filter learning)
3. **Beamforming** (weighted combination of all microphones)

This architecture achieves state-of-the-art performance while maintaining low computational complexity, making it ideal for practical deployment in audio zooming applications where real-time operation on resource-constrained devices is required.

---

## References

1. Deep Audio Zooming: Beamwidth-Controllable Neural Beamformer (Original Paper)
2. AISHELL-2 Speech Corpus
3. DEMAND Noise Database
