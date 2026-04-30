"""Train a tiny annotation-vs-structure crop classifier for PlanVectorizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.architecture.annotation_classifier import (
    AnnotationMLPClassifier,
    load_labeled_dataset,
)


def main() -> int:
    """Train and save the annotation crop classifier."""
    parser = argparse.ArgumentParser(
        description="Train a small MLP on labeled annotation candidate crops.",
    )
    parser.add_argument(
        "dataset_dir",
        help="Directory containing class-named folders of training crop images.",
    )
    parser.add_argument(
        "--output-model",
        default=str(Path("models") / "annotation_classifier.npz"),
        help="Where to save the trained `.npz` model.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=32,
        help="Normalized square crop size for training and inference.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=100,
        help="Hidden neuron count for the MLP.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.12,
        help="Mini-batch gradient descent learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and initialization.",
    )
    args = parser.parse_args()

    features, labels, class_names = load_labeled_dataset(
        dataset_dir=args.dataset_dir,
        input_size=args.input_size,
    )
    train_features, train_labels, validation_features, validation_labels = _split_dataset(
        features,
        labels,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    classifier = AnnotationMLPClassifier.create_random(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        class_names=class_names,
        seed=args.seed,
    )
    classifier.fit(
        train_features,
        train_labels,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    classifier.save(args.output_model)

    print(f"Classes: {', '.join(class_names)}")
    print(f"Training samples: {len(train_features)}")
    print(f"Validation samples: {len(validation_features)}")
    print(f"Training accuracy: {_accuracy(classifier, train_features, train_labels):.2%}")
    if len(validation_features) > 0:
        print(
            "Validation accuracy: "
            f"{_accuracy(classifier, validation_features, validation_labels):.2%}"
        )
    print(f"Saved model: {Path(args.output_model).resolve()}")
    return 0


def _split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split a dataset into train and validation subsets."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(features))
    shuffled_features = features[permutation]
    shuffled_labels = labels[permutation]

    if len(shuffled_features) < 5 or validation_ratio <= 0.0:
        empty_features = np.zeros((0, shuffled_features.shape[1]), dtype=np.float32)
        empty_labels = np.zeros((0,), dtype=np.int64)
        return shuffled_features, shuffled_labels, empty_features, empty_labels

    validation_count = int(round(len(shuffled_features) * validation_ratio))
    validation_count = min(max(1, validation_count), len(shuffled_features) - 1)
    split_index = len(shuffled_features) - validation_count

    return (
        shuffled_features[:split_index],
        shuffled_labels[:split_index],
        shuffled_features[split_index:],
        shuffled_labels[split_index:],
    )


def _accuracy(
    classifier: AnnotationMLPClassifier,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute simple classification accuracy."""
    if len(features) == 0:
        return 0.0

    probabilities = classifier.predict_proba(features)
    predictions = np.argmax(probabilities, axis=1)
    return float(np.mean(predictions == labels))


if __name__ == "__main__":
    raise SystemExit(main())
