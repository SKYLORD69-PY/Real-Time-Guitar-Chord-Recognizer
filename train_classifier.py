"""Multi-class chord classification and model evaluation pipeline.

Loads the Chroma/RMS feature matrix produced by ``extract_chord_features.py``,
compares a Support Vector Machine against a Random Forest under grouped,
label-stratified cross-validation, and serializes the stronger model as a
single self-contained artifact for inference.

Command line
------------
    python train_classifier.py \\
        --features data/processed/chord_features.csv \\
        --output-dir models

Inputs
------
``chord_features.csv`` and its ``*.meta.json`` sidecar, both written by
``extract_chord_features.py``.  The metadata file is the source of truth for
feature column order and names; this module never hard-codes them, so a
classifier trained from any ``FeatureConfig`` stays consistent with the data
it was built from.

Evaluation methodology
-----------------------
Guitar recordings are cut into overlapping analysis segments (see
``extract_chord_features.py``), so many rows in the feature table come from
the same source recording.  A classifier that saw one segment of a recording
during training would have an unfair advantage recognising a neighbouring
segment of the *same* recording at test time: the two share room tone,
microphone placement and exact tuning, none of which a model should be
allowed to use as a shortcut for the chord label.

Two safeguards keep every evaluation honest:

Grouping
    Every split (the outer train/test split and the inner cross-validation
    folds used for hyperparameter search) is computed with
    ``StratifiedGroupKFold``, keyed on ``source_file``.  All segments of one
    recording always land entirely on one side of a split.
Stratification
    ``StratifiedGroupKFold`` additionally balances each class's proportion
    across folds as closely as the grouping constraint allows, which matters
    here because segment counts per class vary with recording length.

Outputs
-------
``<output-dir>/chord_classifier.joblib``
    A dict artifact: the fitted estimator (``pipeline``), ``feature_names`` in
    the exact column order it expects, its ``classes`` in prediction order,
    the ``feature_config`` it was trained against, and training metadata.
    This is the only file ``predict_chord.py`` needs to load.
``<output-dir>/training_report.json``
    Hyperparameter search results and held-out test metrics for every
    candidate model, plus which one was selected and why.
``<output-dir>/classification_report.txt``
    Human-readable per-class precision/recall/F1 for the selected model.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

__all__ = [
    "DEFAULT_RANDOM_STATE",
    "MODEL_CHOICES",
    "CandidateModel",
    "Dataset",
    "TrainingConfig",
    "build_candidates",
    "load_dataset",
    "train_and_select",
]

LOGGER = logging.getLogger("chord_classifier")

DEFAULT_RANDOM_STATE = 42
MIN_GROUPS_PER_CLASS = 2  # smallest number of distinct recordings a class can have
MODEL_CHOICES: tuple[str, ...] = ("svm", "random_forest")
CALIBRATION_FOLDS = 3  # internal folds SVC's probability calibration uses per training fold


@dataclass(frozen=True)
class TrainingConfig:
    """Parameters controlling splitting, search and model selection.

    Attributes:
        models: Which candidate models to train and compare.
        n_outer_splits: Folds for the train/test split; one fold is held out
            as the test set, so e.g. 5 gives an approximate 80/20 split.
            Reduced automatically if a class has fewer source recordings.
        n_inner_splits: Folds for the hyperparameter search's cross-validation,
            run only on the training portion. Reduced the same way.
        scoring: sklearn scorer name used both to pick hyperparameters and to
            select between candidate models. Macro-averaged F1 weighs every
            chord class equally regardless of how many segments it has.
        random_state: Seed for every split and every stochastic estimator.
        n_jobs: Parallel workers for the hyperparameter search.
    """

    models: tuple[str, ...] = MODEL_CHOICES
    n_outer_splits: int = 5
    n_inner_splits: int = 5
    scoring: str = "f1_macro"
    random_state: int = DEFAULT_RANDOM_STATE
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if not self.models or any(m not in MODEL_CHOICES for m in self.models):
            raise ValueError(f"models must be a non-empty subset of {MODEL_CHOICES}")
        if self.n_outer_splits < 2 or self.n_inner_splits < 2:
            raise ValueError("n_outer_splits and n_inner_splits must each be at least 2")


@dataclass(frozen=True)
class Dataset:
    """Feature matrix loaded from an ``extract_chord_features.py`` feature table."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: tuple[str, ...]
    feature_config: dict[str, Any]
    classes: tuple[str, ...]


@dataclass
class CandidateModel:
    """One model family: its unfitted estimator and the grid to search over."""

    name: str
    estimator: BaseEstimator
    param_grid: dict[str, list[Any]]


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def load_dataset(features_path: Path, metadata_path: Path) -> Dataset:
    """Load and validate a feature table written by ``extract_chord_features.py``.

    Raises:
        FileNotFoundError: If either input file is missing.
        ValueError: If required columns are absent, values are non-finite,
            or fewer than two classes are present.
    """
    if not features_path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {features_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path} (expected the *.meta.json written "
            "alongside the feature CSV; pass --metadata explicitly if it was moved)"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = tuple(metadata["feature_names"])
    feature_config = dict(metadata["feature_config"])

    table = pd.read_csv(features_path)
    required = {"label", "source_file", *feature_names}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"Feature CSV is missing {len(missing)} expected column(s): {sorted(missing)[:10]}"
        )
    if table.empty:
        raise ValueError(f"Feature CSV has no rows: {features_path}")

    feature_block = table[list(feature_names)]
    if not np.isfinite(feature_block.to_numpy(dtype=np.float64)).all():
        raise ValueError("Feature CSV contains NaN or infinite values")

    classes = tuple(sorted(table["label"].unique()))
    if len(classes) < 2:
        raise ValueError(f"At least two chord classes are required, found: {classes}")

    return Dataset(
        X=feature_block.to_numpy(dtype=np.float64),
        y=table["label"].to_numpy(dtype=str),
        groups=table["source_file"].to_numpy(dtype=str),
        feature_names=feature_names,
        feature_config=feature_config,
        classes=classes,
    )


def groups_per_class(y: np.ndarray, groups: np.ndarray) -> dict[str, int]:
    """Number of distinct source recordings backing each class label."""
    return {
        str(label): int(pd.Series(groups[y == label]).nunique()) for label in sorted(np.unique(y))
    }


def resolve_split_count(requested: int, groups_by_class: dict[str, int], purpose: str) -> int:
    """Clamp a requested fold count to what the data can actually support.

    Raises:
        ValueError: If the rarest class has fewer than ``MIN_GROUPS_PER_CLASS``
            distinct recordings, since no valid split can then be formed.
    """
    scarcest_label, scarcest_count = min(groups_by_class.items(), key=lambda item: item[1])
    if scarcest_count < MIN_GROUPS_PER_CLASS:
        raise ValueError(
            f"Class '{scarcest_label}' has only {scarcest_count} distinct source recording(s); "
            f"at least {MIN_GROUPS_PER_CLASS} are needed to form {purpose}. Record more takes of "
            "that chord or remove it from the dataset."
        )
    resolved = min(requested, scarcest_count)
    if resolved < requested:
        LOGGER.warning(
            "Reducing %s from %d to %d fold(s): class '%s' has only %d distinct recording(s)",
            purpose,
            requested,
            resolved,
            scarcest_label,
            scarcest_count,
        )
    return resolved


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_train_test(
    dataset: Dataset, n_splits: int, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """One grouped, label-stratified train/test split; returns (train_idx, test_idx)."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(dataset.X, dataset.y, dataset.groups))
    train_files = set(dataset.groups[train_idx])
    test_files = set(dataset.groups[test_idx])
    overlap = train_files & test_files
    if overlap:
        # Defensive only: StratifiedGroupKFold guarantees this cannot happen.
        raise RuntimeError(f"Train/test split leaked {len(overlap)} source file(s) across sides")
    return train_idx, test_idx


# --------------------------------------------------------------------------- #
# Candidate models
# --------------------------------------------------------------------------- #
def build_candidates(config: TrainingConfig) -> list[CandidateModel]:
    """Construct the requested candidate estimators with their search grids.

    SVM (RBF kernel)
        Chroma and RMS statistics are bounded, continuous features where
        chord classes are not linearly separable (e.g. relative-minor chords
        share most of their pitch classes); an RBF kernel captures that
        curved decision boundary. Features are standardized first because
        SVMs are scale-sensitive. ``SVC`` itself only outputs a decision
        function, not class probabilities, so it is wrapped in
        ``CalibratedClassifierCV`` (Platt scaling), which fits a sigmoid on
        held-out folds of the training data to produce the calibrated
        confidence scores ``predict_chord.py`` will report.
    Random Forest
        An ensemble of decision trees that needs no feature scaling and
        models interactions between pitch classes (e.g. "high A and high C#
        but not E" distinguishing a first-fret voicing) without an explicit
        kernel choice.

    Both estimators use ``class_weight="balanced"`` because segment counts
    per chord vary with how long each recording is, not with how common the
    chord should be in the trained model.
    """
    candidates: list[CandidateModel] = []
    if "svm" in config.models:
        candidates.append(
            CandidateModel(
                name="svm",
                estimator=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "svm",
                            CalibratedClassifierCV(
                                SVC(
                                    kernel="rbf",
                                    class_weight="balanced",
                                    random_state=config.random_state,
                                ),
                                method="sigmoid",
                                cv=CALIBRATION_FOLDS,
                                ensemble=False,
                            ),
                        ),
                    ]
                ),
                param_grid={
                    "svm__estimator__C": [1.0, 10.0, 50.0],
                    "svm__estimator__gamma": ["scale", 0.01, 0.1],
                },
            )
        )
    if "random_forest" in config.models:
        candidates.append(
            CandidateModel(
                name="random_forest",
                estimator=RandomForestClassifier(
                    class_weight="balanced", random_state=config.random_state
                ),
                param_grid={
                    "n_estimators": [200, 400],
                    "max_depth": [None, 16],
                    "min_samples_leaf": [1, 3],
                },
            )
        )
    return candidates


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #
def summarize_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...]
) -> dict[str, Any]:
    """Accuracy, macro-F1, full classification report and confusion matrix."""
    report = classification_report(
        y_true, y_pred, labels=list(classes), output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(classes))
    return {
        "accuracy": float(report["accuracy"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(classes),
    }


def fit_candidate(
    candidate: CandidateModel,
    dataset: Dataset,
    train_idx: np.ndarray,
    config: TrainingConfig,
) -> GridSearchCV:
    """Run the grouped, stratified hyperparameter search for one candidate."""
    X_train, y_train, groups_train = (
        dataset.X[train_idx],
        dataset.y[train_idx],
        dataset.groups[train_idx],
    )
    inner_splits = resolve_split_count(
        config.n_inner_splits,
        groups_per_class(y_train, groups_train),
        purpose=f"the inner cross-validation for '{candidate.name}'",
    )
    cv = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=config.random_state)

    search = GridSearchCV(
        estimator=candidate.estimator,
        param_grid=candidate.param_grid,
        scoring=config.scoring,
        cv=cv,
        n_jobs=config.n_jobs,
        refit=True,
    )
    with warnings.catch_warnings():
        # Folds that end up missing a rare class score that class as 0 by
        # zero_division above; sklearn's own warning about it is expected
        # and would otherwise drown out genuine problems in the log.
        warnings.filterwarnings("ignore", message=".*least populated class.*")
        search.fit(X_train, y_train, groups=groups_train)
    return search


def train_and_select(
    dataset: Dataset, config: TrainingConfig
) -> tuple[str, dict[str, Any], dict[str, GridSearchCV]]:
    """Train every candidate, evaluate each on a held-out test split, pick the best.

    Returns:
        ``(best_name, report, searches)`` where ``report`` holds every
        candidate's search and test results plus the selection outcome, and
        ``searches`` maps model name to its fitted ``GridSearchCV`` (the
        winner's ``best_estimator_`` is the artifact that gets saved).
    """
    outer_splits = resolve_split_count(
        config.n_outer_splits,
        groups_per_class(dataset.y, dataset.groups),
        purpose="the train/test split",
    )
    train_idx, test_idx = split_train_test(dataset, outer_splits, config.random_state)
    LOGGER.info(
        "Train/test split: %d training segment(s) from %d recording(s), "
        "%d test segment(s) from %d recording(s)",
        len(train_idx),
        len(set(dataset.groups[train_idx])),
        len(test_idx),
        len(set(dataset.groups[test_idx])),
    )

    candidates = build_candidates(config)
    searches: dict[str, GridSearchCV] = {}
    comparison: dict[str, Any] = {}

    for candidate in candidates:
        LOGGER.info("Searching hyperparameters for '%s'", candidate.name)
        search = fit_candidate(candidate, dataset, train_idx, config)
        searches[candidate.name] = search

        y_pred = search.predict(dataset.X[test_idx])
        test_metrics = summarize_predictions(dataset.y[test_idx], y_pred, dataset.classes)
        comparison[candidate.name] = {
            "best_params": search.best_params_,
            f"cv_{config.scoring}_mean": float(search.best_score_),
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_classification_report": test_metrics["classification_report"],
            "test_confusion_matrix": test_metrics["confusion_matrix"],
        }
        LOGGER.info(
            "  %-13s cv_%s=%.3f  test_accuracy=%.3f  test_f1_macro=%.3f",
            candidate.name,
            config.scoring,
            search.best_score_,
            test_metrics["accuracy"],
            test_metrics["f1_macro"],
        )

    best_name = max(comparison, key=lambda name: comparison[name]["test_f1_macro"])
    report = {
        "scoring": config.scoring,
        "selection_metric": "test_f1_macro",
        "selected_model": best_name,
        "n_train_segments": int(len(train_idx)),
        "n_test_segments": int(len(test_idx)),
        "n_train_recordings": len(set(dataset.groups[train_idx])),
        "n_test_recordings": len(set(dataset.groups[test_idx])),
        "classes": list(dataset.classes),
        "candidates": comparison,
    }
    return best_name, report, searches


# --------------------------------------------------------------------------- #
# Artifact serialization
# --------------------------------------------------------------------------- #
def build_artifact(
    best_name: str,
    searches: dict[str, GridSearchCV],
    dataset: Dataset,
    report: dict[str, Any],
    config: TrainingConfig,
) -> dict[str, Any]:
    """Assemble the self-contained bundle ``predict_chord.py`` will load.

    The winning estimator is refit on the full dataset (train and test
    segments together) with its selected hyperparameters, so the shipped
    model benefits from every labelled recording while the metrics in
    ``training_report.json`` still reflect genuine held-out performance.
    """
    best_estimator = clone(searches[best_name].best_estimator_)
    best_estimator.fit(dataset.X, dataset.y)

    return {
        "model_name": best_name,
        "pipeline": best_estimator,
        "feature_names": list(dataset.feature_names),
        "classes": [str(c) for c in best_estimator.classes_],
        "feature_config": dataset.feature_config,
        "best_params": searches[best_name].best_params_,
        "held_out_test_metrics": {
            "accuracy": report["candidates"][best_name]["test_accuracy"],
            "f1_macro": report["candidates"][best_name]["test_f1_macro"],
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": config.random_state,
        "sklearn_version": package_version("scikit-learn"),
    }


def save_outputs(
    artifact: dict[str, Any],
    report: dict[str, Any],
    dataset: Dataset,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write the model bundle, the full comparison report and a text summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "chord_classifier.joblib"
    joblib.dump(artifact, model_path)

    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # The shipped artifact is refit on every labelled segment (see build_artifact),
    # so this report is unambiguously in-sample for it; the held-out numbers that
    # actually justified picking this model live in training_report.json instead.
    best_name = artifact["model_name"]
    y_pred = artifact["pipeline"].predict(dataset.X)
    text_report = classification_report(
        dataset.y, y_pred, labels=list(dataset.classes), zero_division=0
    )
    classification_path = output_dir / "classification_report.txt"
    classification_path.write_text(
        f"Model: {best_name}\n"
        f"(in-sample report for the final model, fit on all labelled data; see "
        f"training_report.json for the held-out test metrics used to select it)\n\n{text_report}",
        encoding="utf-8",
    )
    return model_path, report_path, classification_path


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare chord classifiers on a Chroma/RMS feature table, "
            "and serialize the better-performing model."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/chord_features.csv"),
        help="Feature CSV written by extract_chord_features.py.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Feature metadata JSON; defaults to --features with a .meta.json suffix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory for the serialized model and training reports.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=list(defaults.models),
        help="Candidate model(s) to train and compare.",
    )
    parser.add_argument("--n-outer-splits", type=int, default=defaults.n_outer_splits)
    parser.add_argument("--n-inner-splits", type=int, default=defaults.n_inner_splits)
    parser.add_argument(
        "--scoring",
        default=defaults.scoring,
        help="sklearn scorer name used for hyperparameter search and model selection.",
    )
    parser.add_argument("--random-state", type=int, default=defaults.random_state)
    parser.add_argument(
        "--jobs",
        type=int,
        default=defaults.n_jobs,
        help="Parallel workers (-1 uses all CPU cores).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        config = TrainingConfig(
            models=tuple(args.models),
            n_outer_splits=args.n_outer_splits,
            n_inner_splits=args.n_inner_splits,
            scoring=args.scoring,
            random_state=args.random_state,
            n_jobs=args.jobs,
        )
    except ValueError as exc:
        parser.error(str(exc))

    metadata_path = args.metadata or args.features.with_suffix(".meta.json")
    try:
        dataset = load_dataset(args.features, metadata_path)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info(
        "Loaded %d segment(s) across %d class(es) from %d source recording(s)",
        len(dataset.y),
        len(dataset.classes),
        len(set(dataset.groups)),
    )
    for label, count in groups_per_class(dataset.y, dataset.groups).items():
        LOGGER.info("  %-12s %5d recording(s)", label, count)

    try:
        best_name, report, searches = train_and_select(dataset, config)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info(
        "Selected '%s' (test accuracy=%.3f, test f1_macro=%.3f)",
        best_name,
        report["candidates"][best_name]["test_accuracy"],
        report["candidates"][best_name]["test_f1_macro"],
    )

    artifact = build_artifact(best_name, searches, dataset, report, config)
    model_path, report_path, classification_path = save_outputs(
        artifact, report, dataset, args.output_dir
    )
    LOGGER.info("Model artifact: %s", model_path)
    LOGGER.info("Training report: %s", report_path)
    LOGGER.info("Classification report: %s", classification_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
