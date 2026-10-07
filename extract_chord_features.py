"""Chroma feature extraction engine for guitar chord recognition.

Converts a labelled folder of guitar chord recordings into a tabular feature
matrix (CSV) for classical machine-learning models such as SVMs and Random
Forests.

Dataset layout
--------------
One sub-folder per chord; the folder name is the class label::

    data/raw/
    ├── Am/   take_01.wav  take_02.wav  ...
    ├── C/    ...
    └── G7/   ...

Command line
------------
    python extract_chord_features.py --dataset-dir data/raw --validate-only
    python extract_chord_features.py --dataset-dir data/raw \\
        --output data/processed/chord_features.csv

Outputs
-------
``chord_features.csv``
    One row per analysis segment: ``label``, ``source_file``, ``segment_index``
    followed by the columns in ``FEATURE_NAMES``.  ``source_file`` identifies
    the recording a segment was cut from.  Use it as the grouping key when
    splitting into train and test sets so that segments from one recording
    never land on both sides of the split (otherwise evaluation scores are
    inflated by near-duplicate samples).
``chord_features.meta.json``
    Extraction parameters, feature names and class counts.  Downstream code
    rebuilds the exact ``FeatureConfig`` from this file with
    ``load_feature_config``.

Reuse
-----
``extract_feature_vector`` is the single implementation of the feature
pipeline.  Batch extraction and any live or single-clip inference must call it
so that a model always sees features computed identically to its training data.

Music and DSP background
------------------------
Pitch classes
    Western music uses 12-tone equal temperament: every octave doubles the
    frequency and is split into 12 equal semitones.  A frequency ``f`` maps to
    a MIDI note number and then to one of 12 pitch classes::

        midi        = 69 + 12 * log2(f / 440)     # A4 = 440 Hz = MIDI 69
        pitch_class = round(midi) mod 12           # 0=C 1=C# 2=D ... 9=A 11=B

        82.4 Hz (E2) -> MIDI 40 -> class 4 (E)
        110 Hz  (A2) -> MIDI 45 -> class 9 (A)
        220 Hz  (A3) -> MIDI 57 -> class 9 (A)

    Notes an octave apart share a pitch class ("octave equivalence").  A chord
    is identified by *which* pitch classes sound together (C major = C, E, G),
    not by the octave or fretboard voicing in which they are played.

Chroma
    A chromagram folds the spectrum onto those 12 classes: the energy at every
    frequency present in the signal is assigned to the pitch class of that
    frequency and summed across all octaves, giving a 12-dimensional pitch
    class profile per analysis frame.  Register and much of the timbre are
    discarded while the harmonic content that defines a chord is kept.

Harmonic leakage
    A plucked string produces partials at 2f, 3f, 4f, ... above its
    fundamental f.  Measured in semitones above the fundamental, partials 1-8
    land on 0, 12, 19, 24, 28, 31, 34 and 36, i.e. on the pitch classes root,
    root, fifth, root, major third, fifth, minor seventh and root.  Overtones
    therefore push energy into pitch classes the player never fretted (the
    7th partial of a C note lands on B-flat, the seventh of a C7 chord).
    Consistent leakage patterns are one reason a trained classifier beats a
    simple "pick the three strongest bins" rule.

Chroma STFT
    Built from a linear-frequency STFT.  librosa's chroma filter bank assigns
    every FFT bin to the pitch class of its centre frequency (after correcting
    for the recording's estimated tuning offset) and applies a Gaussian
    weighting over octaves, centred on 880 Hz with a two-octave width, that
    de-emphasises rumble and very high partials.  Because bins are equally
    spaced in Hz, resolution is coarse where semitones are close together:
    with sr=22050 and n_fft=4096 a bin is 5.4 Hz wide, wider than the 4.9 Hz
    semitone spacing at E2 (82 Hz).  Low fundamentals therefore smear across
    neighbouring classes and the pitch class is carried mostly by their higher
    partials, which is why a long analysis window (n_fft) is used.

Chroma CENS
    "Chroma Energy Normalized Statistics" starts from a constant-Q transform
    whose log-spaced bins give every note the same relative resolution (36
    bins per octave, i.e. 3 per semitone), so the bass strings are resolved
    properly.  The chroma is then (1) L1-normalised per frame, (2) quantised to
    five levels with log-like thresholds (0.05, 0.1, 0.2, 0.4), (3) smoothed
    over time with a Hann window and (4) L2-normalised.  The result is robust
    to dynamics, timbre and articulation, which complements the finer spectral
    detail in Chroma STFT.  The CQT starts at E2 (82.4 Hz), the low E string
    of a standard-tuned guitar, and spans five octaves.  Analysing only the
    guitar range keeps the longest CQT filter near 0.6 s, which is what makes
    buffers of about a second usable.

RMS energy
    Root-mean-square amplitude per frame, ``sqrt(mean(x**2))``, is a loudness
    envelope.  Chroma frames are level-normalised by construction, so RMS adds
    the complementary attack, sustain and decay information and doubles as a
    simple activity measure for rejecting silence.

Segment-level features
    Classifiers need fixed-length input, so each chroma family is summarised
    over the analysis segment by its per-pitch-class mean (the average pitch
    class profile, which carries the chord identity) and standard deviation
    (how stable the harmony is across a strum or arpeggio).  RMS is summarised
    by its mean and standard deviation, giving a 50-dimensional vector:
    2 chroma families x 2 statistics x 12 pitch classes + 2 RMS statistics.

Level invariance
    Each segment is peak-normalised before analysis (``peak_normalize``) so
    recordings made at different gain settings are directly comparable and a
    classifier cannot use recording level as a proxy for the chord label.  RMS
    therefore describes the relative energy profile inside a segment rather
    than absolute loudness.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from importlib.metadata import version as package_version
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

__all__ = [
    "DEFAULT_CONFIG",
    "FEATURE_NAMES",
    "METADATA_COLUMNS",
    "PITCH_CLASSES",
    "FeatureConfig",
    "extract_feature_vector",
    "load_feature_config",
]

LOGGER = logging.getLogger("chord_features")

# The 12 pitch classes of the chromatic scale, indexed 0-11 starting at C.
# This is also the row order of every chromagram librosa returns.
PITCH_CLASSES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)  # fmt: skip

AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".flac", ".ogg", ".mp3"})
METADATA_COLUMNS: tuple[str, ...] = ("label", "source_file", "segment_index")
MIN_RECOMMENDED_FILES_PER_CLASS = 10


def _build_feature_names() -> tuple[str, ...]:
    """Column names in the exact order ``extract_feature_vector`` emits values."""
    names: list[str] = []
    for family in ("chroma_stft", "chroma_cens"):
        for statistic in ("mean", "std"):
            names.extend(f"{family}_{statistic}_{pitch}" for pitch in PITCH_CLASSES)
    names.extend(("rms_mean", "rms_std"))
    return tuple(names)


FEATURE_NAMES: tuple[str, ...] = _build_feature_names()


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters that fully determine the feature vector.

    Training-time extraction and inference must use identical values; the
    configuration is stored alongside the CSV so downstream code can rebuild it.

    Attributes:
        sample_rate: Analysis sample rate in Hz (audio is resampled to this).
        n_fft: STFT window length in samples (frequency resolution).
        hop_length: Samples between successive analysis frames.  Must be a
            multiple of ``2 ** (cqt_n_octaves - 1)`` for the constant-Q transform.
        cqt_fmin_note: Lowest note analysed by the constant-Q transform; E2 is
            the low E string of a standard-tuned guitar.  Lowering it lengthens
            the CQT filters and raises ``min_buffer_seconds`` (C2 needs 1.5 s).
        cqt_n_octaves: Octaves analysed above ``cqt_fmin_note``.
        cens_smoothing_window: Hann smoothing length (frames) for Chroma CENS.
        peak_normalize: Scale every segment to unit peak amplitude.
        trim_silence: Trim leading/trailing silence from each recording.
        trim_top_db: Silence threshold in dB below the recording's peak.
        segment_seconds: Analysis segment length; ``None`` analyses each
            recording as a single segment.  Inference buffers should have this
            same duration because segment statistics depend on it.
        segment_hop_seconds: Hop between consecutive segments of a recording.
        min_buffer_seconds: Shortest buffer analysed.  Shorter recordings or
            live buffers are rejected because the constant-Q transform has too
            few samples for its lowest octave (about 0.75 s at the default range).
        silence_threshold_db: Segments whose RMS is below this level (dBFS,
            measured before peak normalisation) are discarded as silence.
    """

    sample_rate: int = 22_050
    n_fft: int = 4_096
    hop_length: int = 512
    cqt_fmin_note: str = "E2"
    cqt_n_octaves: int = 5
    cens_smoothing_window: int = 41
    peak_normalize: bool = True
    trim_silence: bool = True
    trim_top_db: float = 35.0
    segment_seconds: float | None = 1.5
    segment_hop_seconds: float = 0.5
    min_buffer_seconds: float = 0.75
    silence_threshold_db: float = -50.0

    def __post_init__(self) -> None:
        if min(self.sample_rate, self.n_fft, self.hop_length) <= 0:
            raise ValueError("sample_rate, n_fft and hop_length must be positive")
        if self.cqt_n_octaves < 1:
            raise ValueError("cqt_n_octaves must be at least 1")
        if self.hop_length % (2 ** (self.cqt_n_octaves - 1)):
            raise ValueError(
                f"hop_length={self.hop_length} must be a multiple of "
                f"2**(cqt_n_octaves-1)={2 ** (self.cqt_n_octaves - 1)}"
            )
        if librosa.note_to_hz(self.cqt_fmin_note) * 2**self.cqt_n_octaves >= self.sample_rate / 2:
            raise ValueError(
                f"CQT range {self.cqt_fmin_note} + {self.cqt_n_octaves} octaves exceeds the "
                f"Nyquist frequency at sample_rate={self.sample_rate}"
            )
        if self.segment_hop_seconds <= 0 or self.min_buffer_seconds <= 0:
            raise ValueError("segment_hop_seconds and min_buffer_seconds must be positive")
        if self.segment_seconds is not None and self.segment_seconds < self.min_buffer_seconds:
            raise ValueError("segment_seconds must be None or at least min_buffer_seconds")

    @property
    def min_buffer_samples(self) -> int:
        """Shortest buffer, in samples, that ``extract_feature_vector`` accepts."""
        return max(self.n_fft, math.ceil(self.min_buffer_seconds * self.sample_rate))


DEFAULT_CONFIG = FeatureConfig()


def load_feature_config(metadata_path: str | Path) -> FeatureConfig:
    """Rebuild the ``FeatureConfig`` recorded in a ``*.meta.json`` file."""
    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    known = {item.name for item in fields(FeatureConfig)}
    return FeatureConfig(**{k: v for k, v in payload["feature_config"].items() if k in known})


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #
def extract_feature_vector(
    y: np.ndarray, sr: int, config: FeatureConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Compute the feature vector for one mono audio buffer.

    Args:
        y: Mono waveform of shape ``(n_samples,)``, floats in ``[-1, 1]``.
        sr: Sample rate of ``y`` in Hz; resampled to ``config.sample_rate``.
        config: Feature extraction parameters.

    Returns:
        1-D float64 array aligned with ``FEATURE_NAMES``.

    Raises:
        ValueError: If the buffer is not 1-D, contains NaN/inf, is shorter than
            ``config.min_buffer_seconds``, or yields non-finite features.
    """
    waveform = np.asarray(y, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"expected a mono waveform of shape (n_samples,), got {waveform.shape}")
    if not np.isfinite(waveform).all():
        raise ValueError("waveform contains NaN or infinite values")
    if sr != config.sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=config.sample_rate)
    if waveform.size < config.min_buffer_samples:
        raise ValueError(
            f"buffer too short: {waveform.size / config.sample_rate:.2f} s, "
            f"need at least {config.min_buffer_seconds:.2f} s"
        )
    if config.peak_normalize:
        waveform = librosa.util.normalize(waveform, norm=np.inf)

    # Chroma STFT: linear-frequency spectrum folded onto 12 pitch classes.
    # tuning=None (default) estimates the deviation from A440 for this buffer,
    # so a guitar tuned slightly sharp or flat still maps to the right classes.
    chroma_stft = librosa.feature.chroma_stft(
        y=waveform, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length
    )

    # Chroma CENS: constant-Q chroma, L1-normalised, quantised, time-smoothed.
    chroma_cens = librosa.feature.chroma_cens(
        y=waveform,
        sr=config.sample_rate,
        hop_length=config.hop_length,
        fmin=librosa.note_to_hz(config.cqt_fmin_note),
        n_octaves=config.cqt_n_octaves,
        win_len_smooth=config.cens_smoothing_window,
    )

    # RMS energy per frame: shape (1, n_frames).
    rms = librosa.feature.rms(y=waveform, frame_length=config.n_fft, hop_length=config.hop_length)

    # Collapse the time axis: (12, n_frames) -> mean and std per pitch class.
    vector = np.concatenate(
        [
            chroma_stft.mean(axis=1),
            chroma_stft.std(axis=1),
            chroma_cens.mean(axis=1),
            chroma_cens.std(axis=1),
            rms.mean(axis=1),
            rms.std(axis=1),
        ]
    ).astype(np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("feature extraction produced non-finite values")
    return vector


def rms_dbfs(y: np.ndarray) -> float:
    """Overall RMS level of a waveform in dB relative to full scale."""
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return 20.0 * float(np.log10(max(rms, 1e-12)))


def iter_segments(
    y: np.ndarray, sr: int, segment_seconds: float | None, hop_seconds: float
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(segment_index, samples)`` sliding-window segments of a recording.

    Recordings no longer than one segment are yielded whole.  A trailing
    remainder shorter than a full segment is dropped so that every segment of
    a long recording has the same duration.
    """
    if segment_seconds is None:
        yield 0, y
        return
    segment_len = int(round(segment_seconds * sr))
    hop_len = max(1, int(round(hop_seconds * sr)))
    if y.size <= segment_len:
        yield 0, y
        return
    for index in range(1 + (y.size - segment_len) // hop_len):
        start = index * hop_len
        yield index, y[start : start + segment_len]


# --------------------------------------------------------------------------- #
# Dataset handling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AudioSample:
    """A labelled audio file discovered in the dataset."""

    path: Path
    label: str
    relative_path: str


@dataclass
class ClipResult:
    """Outcome of processing one audio file."""

    sample: AudioSample
    segments: list[tuple[int, np.ndarray]] = field(default_factory=list)
    skipped_segments: int = 0
    error: str | None = None


@dataclass
class ExtractionReport:
    """Feature table plus bookkeeping for a batch run."""

    table: pd.DataFrame
    failures: list[ClipResult]
    skipped_segments: int


def discover_dataset(dataset_dir: Path) -> list[AudioSample]:
    """Collect labelled audio files from ``dataset_dir/<label>/**``.

    Hidden and underscore-prefixed folders (``.ipynb_checkpoints``,
    ``__MACOSX``) are ignored.  Audio files placed directly in ``dataset_dir``
    have no label and are ignored with a warning.

    Raises:
        FileNotFoundError: If ``dataset_dir`` does not exist.
    """
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir} "
            "(expected one sub-folder of audio files per chord label)"
        )

    unlabelled = [
        p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if unlabelled:
        LOGGER.warning(
            "Ignoring %d audio file(s) in the dataset root; labels come from sub-folders",
            len(unlabelled),
        )

    samples: list[AudioSample] = []
    label_dirs = sorted(
        p for p in dataset_dir.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
    )
    for label_dir in label_dirs:
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                samples.append(
                    AudioSample(
                        path=path,
                        label=label_dir.name,
                        relative_path=path.relative_to(dataset_dir).as_posix(),
                    )
                )
    return samples


def summarize_dataset(samples: Sequence[AudioSample]) -> dict[str, int]:
    """Number of audio files per class label, sorted by label."""
    return dict(sorted(Counter(sample.label for sample in samples).items()))


def process_file(sample: AudioSample, config: FeatureConfig) -> ClipResult:
    """Load, clean, segment and featurise one recording.

    Never raises: any failure is captured in ``ClipResult.error`` so a single
    unreadable file cannot abort a batch run.
    """
    try:
        y, _ = librosa.load(sample.path, sr=config.sample_rate, mono=True)
        if config.trim_silence:
            y, _ = librosa.effects.trim(y, top_db=config.trim_top_db)

        if y.size < config.min_buffer_samples:
            return ClipResult(
                sample,
                error=(
                    f"clip too short after preprocessing ({y.size / config.sample_rate:.2f} s, "
                    f"minimum {config.min_buffer_seconds:.2f} s)"
                ),
            )

        segments: list[tuple[int, np.ndarray]] = []
        skipped = 0
        for index, window in iter_segments(
            y, config.sample_rate, config.segment_seconds, config.segment_hop_seconds
        ):
            if rms_dbfs(window) < config.silence_threshold_db:
                skipped += 1
                continue
            segments.append((index, extract_feature_vector(window, config.sample_rate, config)))

        if not segments:
            return ClipResult(
                sample, skipped_segments=skipped, error="no segment above the silence threshold"
            )
        return ClipResult(sample, segments, skipped)
    except Exception as exc:  # noqa: BLE001 - isolate per-file failures from the batch
        return ClipResult(sample, error=f"{type(exc).__name__}: {exc}")


def build_feature_table(
    samples: Sequence[AudioSample], config: FeatureConfig, n_jobs: int = -1
) -> ExtractionReport:
    """Extract features for every sample, in parallel, preserving sample order."""
    workers = Parallel(n_jobs=n_jobs, return_as="generator")
    results = workers(delayed(process_file)(sample, config) for sample in samples)

    meta_rows: list[tuple[str, str, int]] = []
    vectors: list[np.ndarray] = []
    failures: list[ClipResult] = []
    skipped_segments = 0

    with logging_redirect_tqdm():
        for result in tqdm(results, total=len(samples), desc="Extracting features", unit="file"):
            skipped_segments += result.skipped_segments
            if result.error is not None:
                failures.append(result)
                LOGGER.warning("Skipped %s: %s", result.sample.relative_path, result.error)
                continue
            for index, vector in result.segments:
                meta_rows.append((result.sample.label, result.sample.relative_path, index))
                vectors.append(vector)

    if not vectors:
        empty = pd.DataFrame(columns=[*METADATA_COLUMNS, *FEATURE_NAMES])
        return ExtractionReport(empty, failures, skipped_segments)

    table = pd.concat(
        [
            pd.DataFrame(meta_rows, columns=list(METADATA_COLUMNS)),
            pd.DataFrame(np.vstack(vectors), columns=list(FEATURE_NAMES)),
        ],
        axis=1,
    )
    return ExtractionReport(table, failures, skipped_segments)


def write_outputs(table: pd.DataFrame, config: FeatureConfig, output_path: Path) -> Path:
    """Write the feature CSV and its ``.meta.json`` sidecar; return the sidecar path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, float_format="%.6f")

    metadata_path = output_path.with_suffix(".meta.json")
    rows_per_class = table["label"].value_counts().sort_index()
    metadata = {
        "feature_config": asdict(config),
        "feature_names": list(FEATURE_NAMES),
        "pitch_classes": list(PITCH_CLASSES),
        "n_rows": int(len(table)),
        "n_source_files": int(table["source_file"].nunique()),
        "rows_per_class": {str(label): int(count) for label, count in rows_per_class.items()},
        "librosa_version": package_version("librosa"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    defaults = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(
        description=(
            "Extract Chroma STFT, Chroma CENS and RMS features from labelled guitar "
            "chord recordings into a CSV feature matrix."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root folder with one sub-folder of audio files per chord label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chord_features.csv"),
        help="Destination CSV; a .meta.json sidecar is written alongside it.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=defaults.segment_seconds,
        help="Analysis segment length in seconds; 0 analyses each file as one segment.",
    )
    parser.add_argument(
        "--segment-hop-seconds",
        type=float,
        default=defaults.segment_hop_seconds,
        help="Hop between consecutive segments of a file, in seconds.",
    )
    parser.add_argument("--sample-rate", type=int, default=defaults.sample_rate)
    parser.add_argument("--n-fft", type=int, default=defaults.n_fft)
    parser.add_argument("--hop-length", type=int, default=defaults.hop_length)
    parser.add_argument(
        "--trim-silence",
        action=argparse.BooleanOptionalAction,
        default=defaults.trim_silence,
        help="Trim leading/trailing silence from each recording.",
    )
    parser.add_argument("--trim-top-db", type=float, default=defaults.trim_top_db)
    parser.add_argument(
        "--peak-normalize",
        action=argparse.BooleanOptionalAction,
        default=defaults.peak_normalize,
        help="Peak-normalise each segment before analysis.",
    )
    parser.add_argument(
        "--jobs", type=int, default=-1, help="Parallel workers (-1 uses all CPU cores)."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Scan the dataset and report class counts without extracting features.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("numba").setLevel(logging.WARNING)


def log_dataset_summary(counts: dict[str, int]) -> None:
    LOGGER.info("Found %d audio file(s) across %d class(es)", sum(counts.values()), len(counts))
    for label, count in counts.items():
        level = logging.WARNING if count < MIN_RECOMMENDED_FILES_PER_CLASS else logging.INFO
        note = (
            f"  (fewer than {MIN_RECOMMENDED_FILES_PER_CLASS} files)"
            if level == logging.WARNING
            else ""
        )
        LOGGER.log(level, "  %-12s %5d file(s)%s", label, count, note)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        config = replace(
            DEFAULT_CONFIG,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            trim_silence=args.trim_silence,
            trim_top_db=args.trim_top_db,
            peak_normalize=args.peak_normalize,
            segment_seconds=args.segment_seconds if args.segment_seconds > 0 else None,
            segment_hop_seconds=args.segment_hop_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        samples = discover_dataset(args.dataset_dir)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 1

    counts = summarize_dataset(samples)
    if not counts:
        LOGGER.error(
            "No audio files (%s) found under %s",
            ", ".join(sorted(AUDIO_EXTENSIONS)),
            args.dataset_dir,
        )
        return 1
    log_dataset_summary(counts)
    if len(counts) < 2:
        LOGGER.error("At least two chord classes are required for classification")
        return 1
    if args.validate_only:
        LOGGER.info("Dataset validation complete")
        return 0

    report = build_feature_table(samples, config, n_jobs=args.jobs)
    if report.table.empty:
        LOGGER.error("No features could be extracted from any file")
        return 1

    metadata_path = write_outputs(report.table, config, args.output)
    LOGGER.info(
        "Extracted %d segment(s) from %d file(s); %d file(s) skipped, %d silent segment(s) dropped",
        len(report.table),
        len(samples) - len(report.failures),
        len(report.failures),
        report.skipped_segments,
    )
    for label, count in report.table["label"].value_counts().sort_index().items():
        LOGGER.info("  %-12s %5d segment(s)", label, count)
    LOGGER.info("Feature matrix: %s  (%d features)", args.output, len(FEATURE_NAMES))
    LOGGER.info("Extraction metadata: %s", metadata_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
