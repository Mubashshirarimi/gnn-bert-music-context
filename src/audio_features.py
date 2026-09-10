"""Audio loading, log-mel / chroma extraction, and fixed-window segmentation.

Deliberately depends only on torch + torchaudio + soundfile. librosa is avoided
because it pulls in numba, which lags new Python releases (this project runs on
Python 3.14). The chroma filterbank below is a direct reimplementation of
librosa's ``chroma_stft`` construction, so features are comparable.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio

A440 = 440.0
N_PITCH_CLASSES = 12


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Load an audio file as a mono float32 waveform resampled to ``sample_rate``.

    Returns a 1-D tensor. Raises RuntimeError if the file cannot be decoded.
    """
    path = Path(path)
    try:
        waveform, sr = torchaudio.load(str(path))
    except Exception:
        # torchaudio's MP3 support on Windows depends on the ffmpeg backend;
        # soundfile (libsndfile >= 1.1) decodes MP3 without it.
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as exc:
            raise RuntimeError(f"Could not decode {path}: {exc}") from exc
        waveform = torch.from_numpy(data.T.copy())

    if waveform.numel() == 0:
        raise RuntimeError(f"Empty audio: {path}")

    if waveform.shape[0] > 1:  # downmix to mono
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    return waveform.squeeze(0).float()


# --------------------------------------------------------------------------- #
# Spectral features
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    """Computes log-mel and chroma spectrograms with a reusable filterbank cache."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_chroma: int = N_PITCH_CLASSES,
        fmin: float = 30.0,
        fmax: float | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = torch.device(device)

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax or sample_rate / 2,
            power=2.0,
        ).to(self.device)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(self.device)
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=2.0
        ).to(self.device)
        self._chroma_fb = _chroma_filterbank(sample_rate, n_fft, n_chroma).to(self.device)

    def log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """(T,) waveform -> (n_mels, frames) log-power mel spectrogram."""
        with torch.no_grad():
            spec = self.mel(waveform.to(self.device))
            return self.to_db(spec)

    def chroma(self, waveform: torch.Tensor) -> torch.Tensor:
        """(T,) waveform -> (12, frames) L1-normalised chromagram."""
        with torch.no_grad():
            spec = self.spectrogram(waveform.to(self.device))  # (freq, frames)
            chroma = self._chroma_fb @ spec                    # (12, frames)
            # Per-frame L1 normalisation makes chroma comparable across loudness.
            chroma = chroma / chroma.sum(dim=0, keepdim=True).clamp_min(1e-8)
            return chroma


def _chroma_filterbank(sample_rate: int, n_fft: int, n_chroma: int = N_PITCH_CLASSES) -> torch.Tensor:
    """Build a (n_chroma, 1 + n_fft // 2) matrix mapping FFT bins to pitch classes.

    Each FFT bin is assigned to the pitch class of its frequency, weighted by a
    Gaussian over the distance (in semitones) to that pitch class centre.
    """
    freqs = np.linspace(0, sample_rate / 2, 1 + n_fft // 2, endpoint=True)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # avoid log(0) at DC

    # Fractional MIDI-like pitch of every bin, in units of chroma bins.
    tuning = n_chroma * np.log2(freqs / A440)
    bin_pitch = tuning - np.round(tuning / n_chroma) * n_chroma  # wrap to [-6, 6)

    # Distance from each bin to each pitch-class centre, wrapped circularly.
    centres = np.arange(n_chroma, dtype=float)[:, None] - n_chroma / 2.0
    dist = bin_pitch[None, :] - centres
    dist = np.mod(dist + n_chroma / 2.0, n_chroma) - n_chroma / 2.0

    # Gaussian window; sigma of 1 semitone gives moderate pitch-class smearing.
    fb = np.exp(-0.5 * (dist / 1.0) ** 2)

    # Roll so index 0 corresponds to pitch class C rather than A.
    fb = np.roll(fb, -3, axis=0)

    # Taper very low and very high frequencies, which carry little pitch info.
    octave_weight = np.exp(-0.5 * ((np.log2(freqs / A440) - 0.0) / 2.0) ** 2)
    fb = fb * octave_weight[None, :]

    fb = fb / np.maximum(np.linalg.norm(fb, axis=0, keepdims=True), 1e-8)
    return torch.from_numpy(fb).float()


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def segment_features(
    features: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    segment_seconds: float,
    overlap: float = 0.5,
    max_segments: int | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Split a (F, frames) feature matrix into overlapping fixed-length windows.

    Returns
    -------
    segments : (n_segments, F, frames_per_segment)
    times    : (n_segments,) start time of each segment in seconds
    """
    n_frames = features.shape[1]
    frames_per_segment = max(1, int(round(segment_seconds * sample_rate / hop_length)))
    step = max(1, int(round(frames_per_segment * (1.0 - overlap))))

    if n_frames < frames_per_segment:
        # Short clip: pad once up to a single full window.
        pad = frames_per_segment - n_frames
        features = torch.nn.functional.pad(features, (0, pad), mode="replicate")
        n_frames = features.shape[1]

    starts = list(range(0, n_frames - frames_per_segment + 1, step))
    if max_segments is not None and len(starts) > max_segments:
        # Subsample uniformly across the track rather than truncating the tail,
        # so the graph still spans the whole song.
        idx = np.linspace(0, len(starts) - 1, max_segments).round().astype(int)
        starts = [starts[i] for i in idx]

    segments = torch.stack([features[:, s : s + frames_per_segment] for s in starts])
    times = np.array(starts, dtype=float) * hop_length / sample_rate
    return segments, times


def pool_segments(segments: torch.Tensor) -> torch.Tensor:
    """(n_seg, F, frames) -> (n_seg, 2F) mean+std pooled node features."""
    mean = segments.mean(dim=2)
    std = segments.std(dim=2, unbiased=False)
    return torch.cat([mean, std], dim=1)


# --------------------------------------------------------------------------- #
# Chord estimation (for the chord-transition graph)
# --------------------------------------------------------------------------- #
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _chord_templates() -> tuple[torch.Tensor, list[str]]:
    """24 binary major/minor triad templates plus one 'N' (no-chord) template."""
    templates, names = [], []
    major = [0, 4, 7]
    minor = [0, 3, 7]
    for root in range(N_PITCH_CLASSES):
        for intervals, quality in ((major, "maj"), (minor, "min")):
            vec = torch.zeros(N_PITCH_CLASSES)
            for interval in intervals:
                vec[(root + interval) % N_PITCH_CLASSES] = 1.0
            templates.append(vec / vec.norm())
            names.append(f"{PITCH_NAMES[root]}:{quality}")
    return torch.stack(templates), names


CHORD_TEMPLATES, CHORD_NAMES = _chord_templates()


def estimate_chords(chroma: torch.Tensor, min_frames: int = 4) -> list[str]:
    """Frame-wise template matching, then median smoothing and run-length collapse.

    Returns the sequence of chord labels in order of occurrence (repeats collapsed),
    which is exactly what the chord-transition graph needs.
    """
    chroma = chroma.detach().cpu()
    if chroma.shape[1] == 0:
        return []

    scores = CHORD_TEMPLATES @ chroma  # (24, frames)
    best = scores.argmax(dim=0).numpy()

    # Median filter suppresses single-frame flicker between related triads.
    if min_frames > 1 and len(best) >= min_frames:
        kernel = min_frames if min_frames % 2 == 1 else min_frames + 1
        padded = np.pad(best, kernel // 2, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best = np.median(windows, axis=1).astype(int)

    # Collapse consecutive duplicates into a chord sequence.
    sequence: list[str] = []
    previous = -1
    for idx in best:
        if idx != previous:
            sequence.append(CHORD_NAMES[int(idx)])
            previous = int(idx)
    return sequence
