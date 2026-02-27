# Dataset Generation for Deep Audio Zooming

This repository contains the code and instructions to generate a multi-channel audio dataset for training a **beamwidth-controllable neural beamformer**, as described in the paper  
*"Deep Audio Zooming: Beamwidth-Controllable Neural Beamformer"*.

The dataset simulates an 8-channel microphone array in various reverberant rooms, with multiple simultaneous speakers and additive environmental noise. For each utterance, a random field of view (FOV) is sampled, and the target signal is the anechoic sum of all speakers inside that FOV.

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
- [License](#license)

---

## Overview

- **Speech source**: [AISHELL-2](http://www.aishelltech.com/aishell_2) – a large-scale Mandarin speech corpus (16 kHz).
- **Noise source**: [DEMAND](https://zenodo.org/record/1227121) – a database of 16 environmental noises (16 kHz).
- **Room simulation**: `pyroomacoustics` (CPU-based) – optionally `gpuRIR` for GPU acceleration.
- **Microphone array**: 8-channel circular array with 5 cm radius.
- **Output**: For each generated utterance, we store:
  - 8-channel reverberant mixture (with noise)
  - Mono target signal (anechoic sum of speakers inside the sampled FOV)
  - JSON metadata containing all parameters (source positions, room dimensions, T60, SNR, FOV, etc.)

---

## Requirements

Install the required Python packages:

```bash
pip install numpy scipy librosa soundfile pyroomacoustics tqdm
```

Optionally, for faster RIR generation on GPU:

```bash
pip install gpuRIR cupy-cuda12x   # adjust CUDA version to match your system
```

---

## Data Sources

### AISHELL-2

- Download from http://www.aishelltech.com/aishell_2 (registration required).
- Place the `wav` folder (containing `train`, `dev`, `test` subdirectories) in a known location, e.g., `/path/to/aishell-2/wav`.
- The script recursively finds all `.wav` files.

### DEMAND

- Download from https://zenodo.org/record/1227121.
- Place the extracted folder (containing subdirectories like `DKITCHEN`, `DLIVING`, etc.) in a known location, e.g., `/path/to/demand`.
- The script uses any `.wav` file found. If no noise files are provided, the script falls back to generated white noise.

---

## Simulation Parameters

The following parameters are randomised for each generated utterance to ensure diversity:

| Parameter | Range / Values | Description |
|---|---|---|
| Room dimensions | width/depth: 4–8 m, height: 2.5–4 m | Shoebox room size |
| Array position | random with 0.5 m margin from walls | Centre of the microphone array |
| Number of speakers | 1–5 | Each speaker uses a different utterance |
| Source positions | random, at least 0.5 m from array | 3D coordinates inside the room |
| T60 | 0.3–1.3 s | Reverberation time (controlled by `max_order` in `pyroomacoustics`) |
| SNR | 10–40 dB | Signal-to-noise ratio (relative to mixture without noise) |
| FOV (azimuth) | width: 30°–180°, centre random | The angular range that defines the target region |

---

## Generation Process

For each utterance, the following steps are executed:

1. **Select speakers**  
   Randomly choose 1–5 different AISHELL-2 utterances. Load each file, resample to 16 kHz, and pad them to the same length.

2. **Create a room**  
   Randomly generate room dimensions and array position. Place the 8-channel circular microphone array at that position (microphones arranged in a circle of radius 5 cm).

3. **Position sources**  
   Assign a random location `(x, y, z)` to each speaker, ensuring a minimum distance of 0.5 m from the array.

4. **Compute room impulse responses (RIRs)**  
   For each speaker, compute the RIR using `pyroomacoustics` (image-source method, `max_order=15`). Convolve the dry speech signal with its RIR and sum all contributions to obtain the clean multi-channel mixture (without noise).

5. **Add noise**
   - Pick a random noise clip from DEMAND (or generate white noise if DEMAND is unavailable).
   - Place the noise source at a random position in the room.
   - Compute its RIR and convolve with the noise clip.
   - Scale the noise to achieve the desired SNR (based on the clean mixture’s power) and add it to the mixture.

6. **Sample a field of view (FOV)**  
   Randomly choose an azimuth range with width between 30° and 180°, and a random centre. (Elevation is kept fixed at 0 for simplicity; the code can be extended to 3D FOVs.)  
   Determine which speakers lie inside this FOV. If none, resample up to 10 times; after that, fall back to the first speaker.

7. **Build the target signal**  
   The target is the anechoic sum of the dry speech signals of all speakers inside the FOV. This clean signal serves as the learning target for the neural model.

8. **Save the data**
   - Mixture: 8-channel `.wav` file
   - Target: mono `.wav` file
   - Metadata: JSON file containing all simulation parameters (see Output Structure).

---

## Output Structure

The generated dataset is organised as follows:

```
output_dir/
├── train/
│   ├── train_000000_mix.wav
│   ├── train_000000_target.wav
│   ├── train_000000_meta.json
│   ├── train_000001_...
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### Example Metadata JSON

```json
{
  "uid": "train_000000",
  "num_speakers": 3,
  "source_files": [
    "/path/to/aishell-2/wav/train/S0001/BAC009S0001W0123.wav",
    "/path/to/aishell-2/wav/train/S0002/BAC009S0002W0456.wav",
    "/path/to/aishell-2/wav/train/S0003/BAC009S0003W0789.wav"
  ],
  "source_positions": [
    [2.3, 3.1, 1.5],
    [4.1, 1.8, 1.2],
    [5.6, 5.2, 1.6]
  ],
  "array_position": [3.0, 4.0, 1.2],
  "room_size": [6.5, 7.2, 3.0],
  "T60": 0.67,
  "snr_db": 23.5,
  "fov_az_min_rad": -0.5,
  "fov_az_max_rad": 0.8,
  "fov_el_min_rad": -0.2,
  "fov_el_max_rad": 0.3,
  "sources_in_fov": [0, 2]
}
```

- `sources_in_fov` lists the indices (0-based) of speakers inside the sampled FOV.
- All angular values are in radians. The array is assumed to look along the `+x` axis (azimuth 0 = front).

---

## Running the Generation Script

A complete Python script `generate_dataset.py` is provided in this repository.  
Before running, adjust the following variables at the top of the script:

```python
SPEECH_ROOT = "/path/to/aishell-2/wav"   # point to the 'wav' folder containing train/dev/test
NOISE_ROOT = "/path/to/demand"           # optional; set to None or empty string to use white noise
OUTPUT_DIR = "/path/to/output"           # where the dataset will be saved

# Number of utterances per split (adjust to your needs)
N_TRAIN = 5000
N_VAL = 500
N_TEST = 100
```

Then run:

```bash
python generate_dataset.py
```

---

## Notes on Large-Scale Generation

- For the full dataset size reported in the paper (95k training, 2.5k validation, 100 test), a single Kaggle session is insufficient due to time and disk constraints.
- Consider using a multi-GPU cloud instance (e.g., AWS EC2 with multiple GPUs) and replace `pyroomacoustics` with `gpuRIR` for faster RIR computation.
- The script can be parallelised by splitting the utterance list across multiple processes/GPUs.

---

## Citation

If you use this dataset generation code, please cite the original paper:

```bibtex
@inproceedings{deepaudiozoom,
  title={Deep Audio Zooming: Beamwidth-Controllable Neural Beamformer},
  author={...},
  booktitle={...},
  year={2021}
}
```

And the datasets:

- AISHELL-2: http://www.aishelltech.com/aishell_2
- DEMAND: https://doi.org/10.5281/zenodo.1227121

---

## License

This code is provided for research purposes. Please respect the licenses of the original datasets (AISHELL-2 and DEMAND) when using them.
