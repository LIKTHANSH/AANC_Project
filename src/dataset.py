"""
Dataset module for the AANC project.
Handles VCTK corpus extraction, audio loading, preprocessing, and PyTorch Dataset creation.
"""
import os
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import soundfile as sf
import librosa
from src.config import (
    VCTK_ZIP, VCTK_EXTRACTED, DATA_DIR, FEATURES_DIR,
    SAMPLE_RATE, MAX_AUDIO_LENGTH_SEC, MAX_FRAMES,
    SELECTED_SPEAKERS, MAX_UTTERANCES_PER_SPEAKER,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    BATCH_SIZE, NUM_WORKERS, DEVICE
)
from src.features import extract_features


def extract_vctk_corpus():
    """Extract VCTK corpus from zip if not already extracted."""
    wav_dir = os.path.join(VCTK_EXTRACTED, "wav48_silence_trimmed")
    
    # Check if all selected speakers exist and are not empty
    all_speakers_exist = True
    if os.path.exists(wav_dir):
        for speaker_id in SELECTED_SPEAKERS:
            sp_dir = os.path.join(wav_dir, speaker_id)
            if not os.path.exists(sp_dir) or len(os.listdir(sp_dir)) == 0:
                all_speakers_exist = False
                break
    else:
        all_speakers_exist = False
        
    if all_speakers_exist:
        n_speakers = len([d for d in os.listdir(wav_dir) if os.path.isdir(os.path.join(wav_dir, d))])
        print(f"[Dataset] VCTK already extracted. Found {n_speakers} speaker directories.")
        return

    print(f"[Dataset] Extracting VCTK corpus from {VCTK_ZIP}...")
    print(f"[Dataset] This may take a while (11GB zip)...")

    # Extract only the wav48_silence_trimmed and speaker-info.txt
    with zipfile.ZipFile(VCTK_ZIP, 'r') as zf:
        members_to_extract = []
        for name in zf.namelist():
            if name.startswith("wav48_silence_trimmed/") or name == "speaker-info.txt":
                # Filter to only selected speakers
                if name.startswith("wav48_silence_trimmed/"):
                    parts = name.split("/")
                    if len(parts) >= 2:
                        speaker_id = parts[1]
                        if speaker_id in SELECTED_SPEAKERS or speaker_id == "":
                            members_to_extract.append(name)
                else:
                    members_to_extract.append(name)

        print(f"[Dataset] Extracting {len(members_to_extract)} files for {len(SELECTED_SPEAKERS)} speakers...")
        for i, member in enumerate(members_to_extract):
            zf.extract(member, VCTK_EXTRACTED)
            if (i + 1) % 500 == 0:
                print(f"[Dataset] Extracted {i+1}/{len(members_to_extract)} files...")

    print(f"[Dataset] Extraction complete!")


def load_audio(filepath, sr=SAMPLE_RATE):
    """Load and preprocess a single audio file."""
    try:
        audio, orig_sr = sf.read(filepath)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample to target sample rate
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # Trim or pad to fixed length
        max_samples = int(MAX_AUDIO_LENGTH_SEC * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')

        return audio
    except Exception as e:
        print(f"[Dataset] Error loading {filepath}: {e}")
        return None


def build_feature_dataset():
    """Build the feature dataset from VCTK audio files."""
    extract_vctk_corpus()

    wav_dir = os.path.join(VCTK_EXTRACTED, "wav48_silence_trimmed")
    features_file = os.path.join(FEATURES_DIR, "features.npz")

    if os.path.exists(features_file):
        print(f"[Dataset] Loading cached features from {features_file}")
        data = np.load(features_file, allow_pickle=True)
        return data['features'], data['labels'], data['speaker_ids'], data['accent_labels']

    all_features = []
    all_labels = []   # accent index
    all_speaker_ids = []
    all_accent_labels = []  # accent name string

    # Build accent lookup
    from src.config import ACCENT_GROUPS
    speaker_to_accent = {}
    accent_to_idx = {}
    idx = 0
    for accent, speakers in ACCENT_GROUPS.items():
        accent_to_idx[accent] = idx
        idx += 1
        for sp in speakers:
            speaker_to_accent[sp] = accent

    total_processed = 0
    total_skipped = 0

    for speaker_id in SELECTED_SPEAKERS:
        speaker_dir = os.path.join(wav_dir, speaker_id)
        if not os.path.exists(speaker_dir):
            print(f"[Dataset] Speaker {speaker_id} not found, skipping...")
            continue

        accent = speaker_to_accent.get(speaker_id, "Unknown")
        accent_idx = accent_to_idx.get(accent, -1)

        # Get audio files (flac format in VCTK 0.92)
        audio_files = sorted([f for f in os.listdir(speaker_dir)
                             if f.endswith('.flac') and '_mic1' in f])[:MAX_UTTERANCES_PER_SPEAKER]

        for audio_file in audio_files:
            filepath = os.path.join(speaker_dir, audio_file)
            audio = load_audio(filepath)

            if audio is None or len(audio) < SAMPLE_RATE:  # Skip very short clips
                total_skipped += 1
                continue

            # Extract features
            feat = extract_features(audio)
            if feat is not None and feat.shape[0] >= 10:  # Need at least 10 frames
                # Pad/truncate to fixed frame count
                if feat.shape[0] > MAX_FRAMES:
                    feat = feat[:MAX_FRAMES]
                elif feat.shape[0] < MAX_FRAMES:
                    feat = np.pad(feat, ((0, MAX_FRAMES - feat.shape[0]), (0, 0)), mode='constant')

                all_features.append(feat)
                all_labels.append(accent_idx)
                all_speaker_ids.append(speaker_id)
                all_accent_labels.append(accent)
                total_processed += 1

                if total_processed % 100 == 0:
                    print(f"[Dataset] Processed {total_processed} utterances "
                          f"(current speaker: {speaker_id}, accent: {accent})...")

    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    speaker_ids = np.array(all_speaker_ids)
    accent_labels = np.array(all_accent_labels)

    print(f"\n[Dataset] Feature dataset built!")
    print(f"[Dataset] Total samples: {len(features)}")
    print(f"[Dataset] Feature shape: {features.shape}")
    print(f"[Dataset] Skipped: {total_skipped}")
    print(f"[Dataset] Accent distribution:")
    for accent in sorted(set(accent_labels)):
        count = sum(1 for a in accent_labels if a == accent)
        print(f"  {accent}: {count} samples")

    # Cache features
    np.savez_compressed(features_file, features=features, labels=labels,
                       speaker_ids=speaker_ids, accent_labels=accent_labels)
    print(f"[Dataset] Features cached to {features_file}")

    return features, labels, speaker_ids, accent_labels


class AANCDataset(Dataset):
    """PyTorch Dataset for AANC features."""

    def __init__(self, features, labels, speaker_ids=None, accent_labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.speaker_ids = speaker_ids
        self.accent_labels = accent_labels

        # Normalize features using z-score normalization
        self.mean = self.features.mean(dim=(0, 1), keepdim=True)
        self.std = self.features.std(dim=(0, 1), keepdim=True) + 1e-8
        self.features = (self.features - self.mean) / self.std

        # Add channel dimension: [batch, 1, frames, features]
        self.features = self.features.unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_normalization_params(self):
        return self.mean, self.std


def get_data_loaders():
    """Build and return train, validation, and test data loaders."""
    features, labels, speaker_ids, accent_labels = build_feature_dataset()

    dataset = AANCDataset(features, labels, speaker_ids, accent_labels)

    # Split dataset
    total = len(dataset)
    train_size = int(TRAIN_SPLIT * total)
    val_size = int(VAL_SPLIT * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\n[Dataset] Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader, dataset
