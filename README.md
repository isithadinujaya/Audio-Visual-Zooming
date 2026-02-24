# Audio-Visual-Zooming
IEEE Signal Processing cup 2026 Task.

# Dataset Generation for Deep Audio Zooming

This repository contains the code and instructions to generate a multi‑channel audio dataset for training a **beamwidth‑controllable neural beamformer**, as described in the paper  
*"Deep Audio Zooming: Beamwidth‑Controllable Neural Beamformer"*.

The dataset simulates an 8‑channel microphone array in various reverberant rooms, with multiple simultaneous speakers and additive environmental noise. For each utterance, a random field of view (FOV) is sampled, and the target signal is the anechoic sum of all speakers inside that FOV.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Data Sources](#data-sources)
- [Simulation Parameters](#simulation-parameters)
- [Generation Process](#generation-process)
- [Output Structure](#output-structure)
- [Running the Generation Script](#running-the-generation-script)
- [Citation](#citation)

---

## Overview

- **Speech source**: [AISHELL‑2](http://www.aishelltech.com/aishell_2) – a large‑scale Mandarin speech corpus (16 kHz).
- **Noise source**: [DEMAND](https://zenodo.org/record/1227121) – a database of 16 environmental noises (16 kHz).
- **Room simulation**: `pyroomacoustics` (CPU‑based) – optionally `gpuRIR` for GPU acceleration.
- **Microphone array**: 8‑channel circular array with 5 cm radius.
- **Output**: For each generated utterance, we store:
  - 8‑channel reverberant mixture (with noise)
  - Mono target signal (anechoic sum of speakers inside the sampled FOV)
  - JSON metadata containing all parameters (source positions, room dimensions, T60, SNR, FOV, etc.)

---

## Requirements

Install the required Python packages:

```bash
pip install numpy scipy librosa soundfile pyroomacoustics tqdm
