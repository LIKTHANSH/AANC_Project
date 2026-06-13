"""
Feature extraction module for the AANC project.
Extracts MFCCs (with deltas), Pitch (F0), and Energy from audio signals.
"""
import numpy as np
import librosa
from src.config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MFCC, N_MELS, USE_DELTA


def extract_mfcc(audio, sr=SAMPLE_RATE):
    """
    Extract MFCC features from audio signal.

    Returns:
        mfcc: ndarray of shape (n_frames, n_mfcc * 3) if USE_DELTA, else (n_frames, n_mfcc)
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )  # shape: (n_mfcc, n_frames)

    if USE_DELTA:
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.vstack([mfcc, delta, delta2])  # (n_mfcc*3, n_frames)

    return mfcc.T  # (n_frames, n_mfcc*3)


def extract_pitch(audio, sr=SAMPLE_RATE):
    """
    Extract pitch (F0) from audio using fast YIN algorithm.

    Returns:
        pitch: ndarray of shape (n_frames, 1)
    """
    f0 = librosa.yin(
        audio, fmin=80.0,
        fmax=400.0,
        sr=sr, frame_length=N_FFT, hop_length=HOP_LENGTH
    )
    # Replace NaN (unvoiced) with 0
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0.reshape(-1, 1)  # (n_frames, 1)


def extract_energy(audio, sr=SAMPLE_RATE):
    """
    Extract RMS energy from audio.

    Returns:
        energy: ndarray of shape (n_frames, 1)
    """
    energy = librosa.feature.rms(
        y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH
    )  # shape: (1, n_frames)
    return energy.T  # (n_frames, 1)


def extract_features(audio, sr=SAMPLE_RATE):
    """
    Extract combined features: MFCCs (+ deltas) + Pitch + Energy.

    Args:
        audio: 1D numpy array of audio samples
        sr: sample rate

    Returns:
        features: ndarray of shape (n_frames, feature_dim)
                  where feature_dim = 39 (MFCCs) + 1 (pitch) + 1 (energy) = 41
    """
    try:
        mfcc = extract_mfcc(audio, sr)      # (n_frames, 39)
        pitch = extract_pitch(audio, sr)     # (n_frames, 1)
        energy = extract_energy(audio, sr)   # (n_frames, 1)

        # Align frame counts (they should be the same, but just in case)
        min_frames = min(mfcc.shape[0], pitch.shape[0], energy.shape[0])
        mfcc = mfcc[:min_frames]
        pitch = pitch[:min_frames]
        energy = energy[:min_frames]

        # Concatenate all features
        features = np.hstack([mfcc, pitch, energy])  # (n_frames, 41)
        return features.astype(np.float32)

    except Exception as e:
        print(f"[Features] Error extracting features: {e}")
        return None


def extract_mel_spectrogram(audio, sr=SAMPLE_RATE):
    """
    Extract mel spectrogram for visualization and audio reconstruction.

    Returns:
        mel_spec: ndarray of shape (n_mels, n_frames)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def reconstruct_audio_from_mel(mel_spec_db, sr=SAMPLE_RATE):
    """
    Reconstruct audio waveform from mel spectrogram using Griffin-Lim.

    Args:
        mel_spec_db: mel spectrogram in dB scale

    Returns:
        audio: 1D numpy array of reconstructed audio
    """
    # Convert back from dB to power
    mel_spec = librosa.db_to_power(mel_spec_db)

    # Invert mel spectrogram to audio
    audio = librosa.feature.inverse.mel_to_audio(
        mel_spec, sr=sr, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_iter=64
    )
    return audio
