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
