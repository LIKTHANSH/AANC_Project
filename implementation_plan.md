# Accent-Aware Neural Codec (AANC) - Implementation Plan

## Project Goal
Build a neural audio compression model using a Convolutional Autoencoder (CAE) that compresses speech while preserving accent, pitch, and prosody. Train and evaluate on the VCTK Corpus (multi-accent English speech). Compare against MP3 and Opus codecs.

## Environment
- **Python 3.10** with **PyTorch 2.11.0+cu128**
- **GPU**: NVIDIA RTX 5050 (CUDA)
- **Available**: numpy, scipy, sklearn, matplotlib, torch
- **Installing**: librosa, soundfile, pesq, pystoi, torchaudio

## Project Structure
```
D:\Projects\AANC_Project\
├── data\
│   └── DS_10283_3443\
│       └── VCTK-Corpus-0.92.zip   (11GB - already downloaded)
├── src\
│   ├── config.py                   [NEW] Configuration/hyperparameters
│   ├── dataset.py                  [NEW] Data loading, extraction, preprocessing
│   ├── features.py                 [NEW] MFCC, Pitch, Energy extraction
│   ├── model.py                    [NEW] Convolutional Autoencoder architecture
│   ├── train.py                    [NEW] Training loop with GPU support
│   ├── evaluate.py                 [NEW] Evaluation metrics (MSE, PSNR, PESQ, STOI, MCD)
│   └── visualize.py                [NEW] Spectrograms, waveforms, feature maps
├── run_pipeline.py                 [NEW] End-to-end pipeline script
├── compress_audio.py               [NEW] Compress any raw audio file
├── results\                        [NEW] Output directory for results/plots
├── models\                         [NEW] Saved model checkpoints
└── Advanced Data Compression...pdf (Project report)
```

## Cleanup
- Delete `aanc_env\` virtual environment (empty, not used, torch is global)

## Pipeline Steps

### 1. Data Extraction & Preprocessing
- Extract VCTK-Corpus-0.92.zip → `data/VCTK-Corpus-0.92/`
- Load .flac audio files from `wav48_silence_trimmed/`
- Resample all audio to 16kHz mono
- Use a subset of speakers (e.g., 20-30 across diverse accents) for manageable training

### 2. Feature Extraction
- **MFCCs**: 13 coefficients per frame (+ delta & delta-delta = 39 total)
- **Pitch (F0)**: Fundamental frequency via librosa.pyin
- **Energy**: RMS energy per frame
- Frame size: 25ms, hop: 10ms, at 16kHz → 400 samples/frame, 160 samples/hop
- Combine into feature matrix: [n_frames × 41] (39 MFCC + 1 pitch + 1 energy)

### 3. Convolutional Autoencoder Architecture
```
Encoder:
  Input → [batch, 1, n_frames, 41]
  Conv2d(1, 32, 3, stride=2, padding=1) + BN + ReLU
  Conv2d(32, 64, 3, stride=2, padding=1) + BN + ReLU  
  Conv2d(64, 128, 3, stride=2, padding=1) + BN + ReLU
  Flatten → Linear → latent_dim (64)

Decoder (mirror):
  Linear → Reshape
  ConvTranspose2d(128, 64, 3, stride=2, padding=1) + BN + ReLU
  ConvTranspose2d(64, 32, 3, stride=2, padding=1) + BN + ReLU
  ConvTranspose2d(32, 1, 3, stride=2, padding=1) + Sigmoid
```

### 4. Training Configuration
- Optimizer: Adam (lr=0.001)
- Loss: MSE (reconstruction loss)
- Batch size: 64
- Epochs: 100
- Train/Val/Test split: 80/10/10
- GPU: RTX 5050 with mixed precision (AMP)

### 5. Evaluation Metrics
- **MSE** (↓): Mean Squared Error between original and reconstructed features
- **PSNR** (↑): Peak Signal-to-Noise Ratio
- **PESQ** (↑): Perceptual Evaluation of Speech Quality (on reconstructed audio)
- **STOI** (↑): Short-Time Objective Intelligibility
- **MCD** (↓): Mel-Cepstral Distortion
- **Compression Ratio** (↑): Original size / compressed size

### 6. Comparison
- Compare AANC against MP3 (128kbps) and Opus (64kbps)
- Generate comparison tables and visualizations

### 7. Visualizations
- Training loss curves
- Original vs reconstructed mel-spectrograms
- Waveform comparisons
- Feature distribution plots
- Per-accent performance breakdown

## Verification
- Run full training pipeline
- Generate all evaluation metrics
- Create comparison plots
- Test compression on unseen audio files
