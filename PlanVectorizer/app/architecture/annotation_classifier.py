"""Optional tiny MLP classifier for annotation-vs-structure crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class AnnotationPrediction:
    """Predicted class label and confidence for one candidate crop."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class AnnotationMLPClassifier:
    """Small NumPy MLP for candidate crop classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        class_names: Sequence[str],
        weight_input_hidden: np.ndarray,
        bias_hidden: np.ndarray,
        weight_hidden_output: np.ndarray,
        bias_output: np.ndarray,
    ) -> None:
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.class_names = tuple(str(class_name) for class_name in class_names)
        self.weight_input_hidden = weight_input_hidden.astype(np.float32)
        self.bias_hidden = bias_hidden.astype(np.float32)
        self.weight_hidden_output = weight_hidden_output.astype(np.float32)
        self.bias_output = bias_output.astype(np.float32)

    @classmethod
    def create_random(
        cls,
        input_size: int,
        hidden_size: int,
        class_names: Sequence[str],
        seed: int = 42,
    ) -> "AnnotationMLPClassifier":
        """Create a randomly initialized classifier."""
        rng = np.random.default_rng(seed)
        feature_count = input_size * input_size
        class_count = len(class_names)
        scale_hidden = np.sqrt(2.0 / max(1.0, float(feature_count)))
        scale_output = np.sqrt(2.0 / max(1.0, float(hidden_size)))

        weight_input_hidden = rng.normal(
            loc=0.0,
            scale=scale_hidden,
            size=(hidden_size, feature_count),
        ).astype(np.float32)
        bias_hidden = np.zeros(hidden_size, dtype=np.float32)
        weight_hidden_output = rng.normal(
            loc=0.0,
            scale=scale_output,
            size=(class_count, hidden_size),
        ).astype(np.float32)
        bias_output = np.zeros(class_count, dtype=np.float32)

        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            class_names=class_names,
            weight_input_hidden=weight_input_hidden,
            bias_hidden=bias_hidden,
            weight_hidden_output=weight_hidden_output,
            bias_output=bias_output,
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 200,
        learning_rate: float = 0.1,
        batch_size: int = 32,
        seed: int = 42,
    ) -> None:
        """Train the MLP with mini-batch gradient descent."""
        feature_matrix = np.asarray(features, dtype=np.float32)
        label_vector = np.asarray(labels, dtype=np.int64)
        if feature_matrix.ndim != 2:
            raise ValueError("features must have shape (sample_count, feature_count)")
        if len(feature_matrix) != len(label_vector):
            raise ValueError("features and labels must have the same length")
        if len(feature_matrix) == 0:
            raise ValueError("training requires at least one sample")

        class_count = len(self.class_names)
        targets = np.eye(class_count, dtype=np.float32)[label_vector]
        rng = np.random.default_rng(seed)

        for _ in range(max(1, int(epochs))):
            permutation = rng.permutation(len(feature_matrix))
            shuffled_features = feature_matrix[permutation]
            shuffled_targets = targets[permutation]

            for start_index in range(0, len(shuffled_features), max(1, int(batch_size))):
                end_index = start_index + max(1, int(batch_size))
                batch_features = shuffled_features[start_index:end_index]
                batch_targets = shuffled_targets[start_index:end_index]

                hidden_linear = batch_features @ self.weight_input_hidden.T + self.bias_hidden
                hidden_activation = _sigmoid(hidden_linear)
                output_linear = hidden_activation @ self.weight_hidden_output.T + self.bias_output
                output_activation = _softmax(output_linear)

                output_delta = (output_activation - batch_targets) / float(len(batch_features))
                hidden_delta = (output_delta @ self.weight_hidden_output) * hidden_activation * (1.0 - hidden_activation)

                grad_output_weight = output_delta.T @ hidden_activation
                grad_output_bias = output_delta.sum(axis=0)
                grad_hidden_weight = hidden_delta.T @ batch_features
                grad_hidden_bias = hidden_delta.sum(axis=0)

                self.weight_hidden_output -= float(learning_rate) * grad_output_weight
                self.bias_output -= float(learning_rate) * grad_output_bias
                self.weight_input_hidden -= float(learning_rate) * grad_hidden_weight
                self.bias_hidden -= float(learning_rate) * grad_hidden_bias

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class probabilities for one or more feature vectors."""
        feature_matrix = np.asarray(features, dtype=np.float32)
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(1, -1)

        hidden_linear = feature_matrix @ self.weight_input_hidden.T + self.bias_hidden
        hidden_activation = _sigmoid(hidden_linear)
        output_linear = hidden_activation @ self.weight_hidden_output.T + self.bias_output
        return _softmax(output_linear)

    def predict_crop(self, crop_mask: np.ndarray) -> AnnotationPrediction:
        """Classify one binary candidate crop."""
        features = normalize_candidate_crop(crop_mask, self.input_size)
        probabilities = self.predict_proba(features)[0]
        best_index = int(np.argmax(probabilities))
        return AnnotationPrediction(
            label=self.class_names[best_index],
            confidence=float(probabilities[best_index]),
            probabilities={
                class_name: float(probabilities[class_index])
                for class_index, class_name in enumerate(self.class_names)
            },
        )

    def save(self, output_path: str) -> None:
        """Serialize the trained model to a NumPy archive."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_file,
            input_size=np.array([self.input_size], dtype=np.int32),
            hidden_size=np.array([self.hidden_size], dtype=np.int32),
            class_names=np.array(self.class_names, dtype=object),
            weight_input_hidden=self.weight_input_hidden,
            bias_hidden=self.bias_hidden,
            weight_hidden_output=self.weight_hidden_output,
            bias_output=self.bias_output,
        )

    @classmethod
    def load(cls, input_path: str) -> "AnnotationMLPClassifier":
        """Load a classifier from a `.npz` file."""
        archive = np.load(input_path, allow_pickle=True)
        return cls(
            input_size=int(archive["input_size"][0]),
            hidden_size=int(archive["hidden_size"][0]),
            class_names=tuple(str(item) for item in archive["class_names"].tolist()),
            weight_input_hidden=np.asarray(archive["weight_input_hidden"], dtype=np.float32),
            bias_hidden=np.asarray(archive["bias_hidden"], dtype=np.float32),
            weight_hidden_output=np.asarray(archive["weight_hidden_output"], dtype=np.float32),
            bias_output=np.asarray(archive["bias_output"], dtype=np.float32),
        )


def normalize_candidate_crop(crop_mask: np.ndarray, input_size: int = 32) -> np.ndarray:
    """Center and resize one binary crop into a normalized square feature vector."""
    candidate_mask = np.asarray(crop_mask, dtype=np.uint8)
    if candidate_mask.ndim != 2:
        raise ValueError("candidate crop must be a 2D grayscale mask")

    nonzero_points = cv2.findNonZero(candidate_mask)
    if nonzero_points is None:
        return np.zeros((1, input_size * input_size), dtype=np.float32)

    x, y, width, height = cv2.boundingRect(nonzero_points)
    glyph = candidate_mask[y : y + height, x : x + width]

    square_size = max(width, height) + 4
    square_canvas = np.zeros((square_size, square_size), dtype=np.uint8)
    offset_x = (square_size - width) // 2
    offset_y = (square_size - height) // 2
    square_canvas[offset_y : offset_y + height, offset_x : offset_x + width] = glyph

    resized = cv2.resize(
        square_canvas,
        (input_size, input_size),
        interpolation=cv2.INTER_AREA,
    )
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, input_size * input_size)


def load_labeled_dataset(dataset_dir: str, input_size: int = 32) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load a crop dataset stored as class-named folders of images."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_directories = sorted(path for path in dataset_path.iterdir() if path.is_dir())
    if not class_directories:
        raise ValueError("Dataset directory must contain at least one class folder")

    features = []
    labels = []
    class_names = tuple(directory.name for directory in class_directories)

    for class_index, class_directory in enumerate(class_directories):
        for image_path in sorted(class_directory.iterdir()):
            if not image_path.is_file():
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            feature_vector = normalize_candidate_crop(binary_image, input_size)
            features.append(feature_vector[0])
            labels.append(class_index)

    if not features:
        raise ValueError("Dataset directory does not contain any readable training images")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        class_names,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid activation."""
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: np.ndarray) -> np.ndarray:
    """Softmax normalized across the last axis."""
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)
