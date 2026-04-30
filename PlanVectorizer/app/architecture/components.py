"""Connected-component filtering for architecture-only extraction."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Optional, Protocol

import cv2
import numpy as np

from app.architecture.annotation_classifier import AnnotationPrediction


@dataclass(frozen=True)
class ComponentFilterSettings:
    """Heuristics for keeping structural black components and rejecting annotations."""

    pillar_min_area: int = 20
    pillar_min_contour_area: float = 36.0
    pillar_max_aspect_ratio: float = 1.7
    pillar_min_fill_ratio: float = 0.52
    pillar_min_rectangularity: float = 0.68
    pillar_max_vertices: int = 6
    circular_pillar_min_area: int = 50
    circular_pillar_max_aspect_ratio: float = 1.35
    circular_pillar_min_circularity: float = 0.68
    long_structure_min_length: int = 24
    slender_structure_max_thickness: int = 8
    arc_like_min_dimension: int = 18
    arc_like_max_aspect_ratio: float = 2.8
    arc_like_max_fill_ratio: float = 0.32
    annotation_candidate_max_area: int = 420
    annotation_candidate_max_dimension: int = 64
    callout_max_dimension: int = 18
    callout_min_dimension: int = 4
    callout_max_aspect_ratio: float = 1.35
    callout_min_circularity: float = 0.55
    callout_min_vertices: int = 6
    callout_max_rectangularity: float = 0.72
    annotation_cluster_gap: int = 14
    structure_cluster_gap: int = 12
    candidate_padding: int = 6
    classifier_confidence_threshold: float = 0.72
    structure_review_reject_threshold: float = 0.9
    classifier_max_crop_dimension: int = 96
    structure_candidate_max_area: int = 2600
    structure_candidate_max_dimension: int = 96
    structure_candidate_min_perimeter: float = 20.0
    door_candidate_min_dimension: int = 12
    door_candidate_max_aspect_ratio: float = 3.4
    door_candidate_min_perimeter: float = 24.0
    door_candidate_max_fill_ratio: float = 0.42


@dataclass(frozen=True)
class CandidateCrop:
    """One ambiguous crop saved for inspection or AI training."""

    cluster_id: int
    x: int
    y: int
    width: int
    height: int
    member_count: int
    image: np.ndarray
    predicted_label: str = "heuristic_annotation_cluster"
    confidence: float = 0.0
    candidate_kind: str = "annotation_review"
    default_decision: str = "reject"
    final_decision: str = "reject"
    decision_source: str = "heuristic"


@dataclass(frozen=True)
class ComponentFilterResult:
    """Masks and candidate crops produced by the component filter stage."""

    structure_mask: np.ndarray
    rejected_symbol_mask: np.ndarray
    candidate_crops: tuple[CandidateCrop, ...]


@dataclass(frozen=True)
class ComponentDescriptor:
    """Shape statistics for a connected component."""

    label_index: int
    x: int
    y: int
    width: int
    height: int
    area: int
    centroid_x: float
    centroid_y: float
    max_dimension: int
    min_dimension: int
    aspect_ratio: float
    fill_ratio: float
    contour_area: float
    contour_rectangularity: float
    perimeter: float
    circularity: float
    approx_vertices: int


class AnnotationCandidateClassifier(Protocol):
    """Minimal protocol for optional annotation crop classifiers."""

    def predict_crop(self, crop_mask: np.ndarray) -> AnnotationPrediction:
        """Predict the class label for one candidate crop."""


def filter_structure_components(
    black_mask_clean: np.ndarray,
    settings: ComponentFilterSettings,
    classifier: Optional[AnnotationCandidateClassifier] = None,
) -> ComponentFilterResult:
    """Keep likely structural components and reject annotation-like clusters."""
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        black_mask_clean,
        connectivity=8,
    )
    descriptors = _build_descriptors(labels, stats, centroids)
    descriptors_by_label = {descriptor.label_index: descriptor for descriptor in descriptors}

    kept_labels: set[int] = set()
    annotation_candidates: list[ComponentDescriptor] = []
    structure_review_candidates: list[ComponentDescriptor] = []

    for descriptor in descriptors:
        if _is_structural_component(descriptor, settings):
            if _is_structure_review_candidate(descriptor, settings):
                structure_review_candidates.append(descriptor)
            else:
                kept_labels.add(descriptor.label_index)
            continue

        if _is_annotation_candidate(descriptor, settings):
            annotation_candidates.append(descriptor)
            continue

        if _is_structure_review_candidate(descriptor, settings):
            structure_review_candidates.append(descriptor)
            continue

        # Keep ambiguous larger components for now to avoid removing true structure.
        kept_labels.add(descriptor.label_index)

    rejected_labels: set[int] = set()
    candidate_crops = []
    next_cluster_id = 1

    next_cluster_id = _process_candidate_groups(
        black_mask_clean=black_mask_clean,
        descriptors_by_label=descriptors_by_label,
        candidates=annotation_candidates,
        cluster_gap=settings.annotation_cluster_gap,
        candidate_kind="annotation_review",
        default_decision="reject",
        classifier=classifier,
        settings=settings,
        kept_labels=kept_labels,
        rejected_labels=rejected_labels,
        candidate_crops=candidate_crops,
        starting_cluster_id=next_cluster_id,
    )
    _process_candidate_groups(
        black_mask_clean=black_mask_clean,
        descriptors_by_label=descriptors_by_label,
        candidates=structure_review_candidates,
        cluster_gap=settings.structure_cluster_gap,
        candidate_kind="structure_review",
        default_decision="keep",
        classifier=classifier,
        settings=settings,
        kept_labels=kept_labels,
        rejected_labels=rejected_labels,
        candidate_crops=candidate_crops,
        starting_cluster_id=next_cluster_id,
    )

    structure_mask = np.zeros_like(black_mask_clean)
    rejected_symbol_mask = np.zeros_like(black_mask_clean)

    for descriptor in descriptors:
        target_mask = structure_mask
        if descriptor.label_index in rejected_labels and descriptor.label_index not in kept_labels:
            target_mask = rejected_symbol_mask

        target_mask[labels == descriptor.label_index] = 255

    return ComponentFilterResult(
        structure_mask=structure_mask,
        rejected_symbol_mask=rejected_symbol_mask,
        candidate_crops=tuple(candidate_crops),
    )


def _build_descriptors(
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
) -> list[ComponentDescriptor]:
    """Collect reusable metrics for each connected component."""
    descriptors: list[ComponentDescriptor] = []

    for label_index in range(1, len(stats)):
        component_mask = np.where(labels == label_index, 255, 0).astype(np.uint8)
        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        max_dimension = max(width, height)
        min_dimension = max(1, min(width, height))
        bbox_area = max(1.0, float(width * height))
        contour_metrics = _component_contour_metrics(component_mask, bbox_area)

        descriptors.append(
            ComponentDescriptor(
                label_index=label_index,
                x=x,
                y=y,
                width=width,
                height=height,
                area=area,
                centroid_x=float(centroids[label_index][0]),
                centroid_y=float(centroids[label_index][1]),
                max_dimension=max_dimension,
                min_dimension=min_dimension,
                aspect_ratio=max_dimension / float(min_dimension),
                fill_ratio=area / bbox_area,
                contour_area=contour_metrics["contour_area"],
                contour_rectangularity=contour_metrics["contour_rectangularity"],
                perimeter=contour_metrics["perimeter"],
                circularity=contour_metrics["circularity"],
                approx_vertices=int(contour_metrics["approx_vertices"]),
            )
        )

    return descriptors


def _is_structural_component(
    descriptor: ComponentDescriptor,
    settings: ComponentFilterSettings,
) -> bool:
    """Return True when the component confidently looks architectural."""
    if _is_rectangular_pillar(descriptor, settings):
        return True

    if _is_circular_pillar(descriptor, settings):
        return True

    is_filled_pillar = (
        descriptor.area >= settings.pillar_min_area
        and descriptor.aspect_ratio <= settings.pillar_max_aspect_ratio
        and descriptor.fill_ratio >= settings.pillar_min_fill_ratio
    )
    if is_filled_pillar:
        return True

    is_long_orthogonal = (
        descriptor.max_dimension >= settings.long_structure_min_length
        and descriptor.min_dimension <= settings.slender_structure_max_thickness
    )
    if is_long_orthogonal:
        return True

    is_arc_like = (
        descriptor.max_dimension >= settings.arc_like_min_dimension
        and descriptor.aspect_ratio <= settings.arc_like_max_aspect_ratio
        and descriptor.fill_ratio <= settings.arc_like_max_fill_ratio
        and descriptor.perimeter >= settings.long_structure_min_length
    )
    if is_arc_like:
        return True

    return False


def _is_rectangular_pillar(
    descriptor: ComponentDescriptor,
    settings: ComponentFilterSettings,
) -> bool:
    """Keep small orthogonal rectangular loops that resemble pillars."""
    return (
        descriptor.contour_area >= settings.pillar_min_contour_area
        and descriptor.aspect_ratio <= settings.pillar_max_aspect_ratio
        and descriptor.contour_rectangularity >= settings.pillar_min_rectangularity
        and descriptor.approx_vertices <= settings.pillar_max_vertices
    )


def _is_circular_pillar(
    descriptor: ComponentDescriptor,
    settings: ComponentFilterSettings,
) -> bool:
    """Keep larger circular closed shapes that are more likely pillars than callouts."""
    return (
        descriptor.area >= settings.circular_pillar_min_area
        and descriptor.aspect_ratio <= settings.circular_pillar_max_aspect_ratio
        and descriptor.circularity >= settings.circular_pillar_min_circularity
    )


def _is_annotation_candidate(
    descriptor: ComponentDescriptor,
    settings: ComponentFilterSettings,
) -> bool:
    """Return True when a component looks like text, a callout, or a small symbol."""
    if _is_small_callout_circle(descriptor, settings):
        return True

    if (
        descriptor.area <= settings.annotation_candidate_max_area
        and descriptor.max_dimension <= settings.annotation_candidate_max_dimension
    ):
        return True

    return False


def _is_structure_review_candidate(
    descriptor: ComponentDescriptor,
    settings: ComponentFilterSettings,
) -> bool:
    """Return True when a small structural-looking component is worth sending to AI."""
    if descriptor.area > settings.structure_candidate_max_area:
        return False

    if descriptor.max_dimension > settings.structure_candidate_max_dimension:
        return False

    if descriptor.perimeter < settings.structure_candidate_min_perimeter:
        return False

    if _is_rectangular_pillar(descriptor, settings) or _is_circular_pillar(descriptor, settings):
        return True

    if _is_door_like_candidate(descriptor, settings):
        return True

    return (
        descriptor.fill_ratio <= 0.55
        and descriptor.aspect_ratio <= 4.5
        and descriptor.min_dimension >= 3
    )


def _is_door_like_candidate(
    descriptor: ComponentDescriptor,
    settings: ComponentFilterSettings,
) -> bool:
    """Identify small arc/jamb-like structural fragments that often get confused with noise."""
    return (
        descriptor.max_dimension >= settings.door_candidate_min_dimension
        and descriptor.aspect_ratio <= settings.door_candidate_max_aspect_ratio
        and descriptor.perimeter >= settings.door_candidate_min_perimeter
        and descriptor.fill_ratio <= settings.door_candidate_max_fill_ratio
        and descriptor.approx_vertices >= 4
    )


def _is_small_callout_circle(
    descriptor: ComponentDescriptor,
    settings: ComponentFilterSettings,
) -> bool:
    """Detect small round callout marks that often accompany room labels."""
    return (
        settings.callout_min_dimension <= descriptor.max_dimension <= settings.callout_max_dimension
        and descriptor.aspect_ratio <= settings.callout_max_aspect_ratio
        and descriptor.circularity >= settings.callout_min_circularity
        and descriptor.approx_vertices >= settings.callout_min_vertices
        and descriptor.contour_rectangularity <= settings.callout_max_rectangularity
    )


def _cluster_candidate_labels(
    candidates: list[ComponentDescriptor],
    gap: int,
) -> list[tuple[int, ...]]:
    """Group nearby candidate components into review clusters."""
    if not candidates:
        return []

    parents = {descriptor.label_index: descriptor.label_index for descriptor in candidates}

    def find(label_index: int) -> int:
        parent = parents[label_index]
        while parent != parents[parent]:
            parents[parent] = parents[parents[parent]]
            parent = parents[parent]
        parents[label_index] = parent
        return parent

    def union(left_label: int, right_label: int) -> None:
        left_root = find(left_label)
        right_root = find(right_label)
        if left_root != right_root:
            parents[right_root] = left_root

    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if _components_are_close(left, right, gap):
                union(left.label_index, right.label_index)

    clusters: dict[int, list[int]] = {}
    for descriptor in candidates:
        root = find(descriptor.label_index)
        clusters.setdefault(root, []).append(descriptor.label_index)

    return [tuple(sorted(label_indices)) for label_indices in clusters.values()]


def _process_candidate_groups(
    black_mask_clean: np.ndarray,
    descriptors_by_label: dict[int, ComponentDescriptor],
    candidates: list[ComponentDescriptor],
    cluster_gap: int,
    candidate_kind: str,
    default_decision: str,
    classifier: Optional[AnnotationCandidateClassifier],
    settings: ComponentFilterSettings,
    kept_labels: set[int],
    rejected_labels: set[int],
    candidate_crops: list[CandidateCrop],
    starting_cluster_id: int,
) -> int:
    """Evaluate grouped AI review candidates and apply keep/reject decisions."""
    cluster_id = starting_cluster_id
    candidate_clusters = _cluster_candidate_labels(candidates, cluster_gap)

    for cluster_labels in candidate_clusters:
        cluster_descriptors = [descriptors_by_label[label_index] for label_index in cluster_labels]
        cluster_crop = _extract_cluster_crop(black_mask_clean, cluster_descriptors, settings)
        cluster_prediction = _predict_annotation_cluster(
            cluster_crop["image"],
            classifier,
            settings,
        )
        final_decision, decision_source = _resolve_cluster_decision(
            prediction=cluster_prediction,
            default_decision=default_decision,
            candidate_kind=candidate_kind,
            settings=settings,
        )

        candidate_crops.append(
            CandidateCrop(
                cluster_id=cluster_id,
                x=cluster_crop["x"],
                y=cluster_crop["y"],
                width=cluster_crop["width"],
                height=cluster_crop["height"],
                member_count=len(cluster_labels),
                image=cluster_crop["image"],
                predicted_label=cluster_prediction.label,
                confidence=cluster_prediction.confidence,
                candidate_kind=candidate_kind,
                default_decision=default_decision,
                final_decision=final_decision,
                decision_source=decision_source,
            )
        )

        if final_decision == "keep":
            kept_labels.update(cluster_labels)
        else:
            rejected_labels.update(cluster_labels)

        cluster_id += 1

    return cluster_id


def _resolve_cluster_decision(
    prediction: AnnotationPrediction,
    default_decision: str,
    candidate_kind: str,
    settings: ComponentFilterSettings,
) -> tuple[str, str]:
    """Blend AI predictions with safe default behavior for each candidate family."""
    if prediction.label.startswith("structure_"):
        return "keep", "classifier"

    if prediction.label.startswith("annotation_"):
        if (
            candidate_kind == "structure_review"
            and prediction.confidence < settings.structure_review_reject_threshold
        ):
            return default_decision, "heuristic"
        return "reject", "classifier"

    return default_decision, "heuristic"


def _components_are_close(
    left: ComponentDescriptor,
    right: ComponentDescriptor,
    gap: int,
) -> bool:
    """Return True when two candidate components are close enough to form one label cluster."""
    left_x1 = left.x - gap
    left_y1 = left.y - gap
    left_x2 = left.x + left.width + gap
    left_y2 = left.y + left.height + gap
    right_x1 = right.x
    right_y1 = right.y
    right_x2 = right.x + right.width
    right_y2 = right.y + right.height

    overlaps = not (
        left_x2 < right_x1
        or right_x2 < left_x1
        or left_y2 < right_y1
        or right_y2 < left_y1
    )
    if overlaps:
        return True

    centroid_dx = left.centroid_x - right.centroid_x
    centroid_dy = left.centroid_y - right.centroid_y
    return centroid_dx * centroid_dx + centroid_dy * centroid_dy <= float((gap * 2) * (gap * 2))


def _extract_cluster_crop(
    black_mask_clean: np.ndarray,
    cluster_descriptors: list[ComponentDescriptor],
    settings: ComponentFilterSettings,
) -> dict[str, object]:
    """Create a padded crop image around one annotation candidate cluster."""
    left = min(descriptor.x for descriptor in cluster_descriptors)
    top = min(descriptor.y for descriptor in cluster_descriptors)
    right = max(descriptor.x + descriptor.width for descriptor in cluster_descriptors)
    bottom = max(descriptor.y + descriptor.height for descriptor in cluster_descriptors)

    left = max(0, left - settings.candidate_padding)
    top = max(0, top - settings.candidate_padding)
    right = min(black_mask_clean.shape[1], right + settings.candidate_padding)
    bottom = min(black_mask_clean.shape[0], bottom + settings.candidate_padding)

    return {
        "x": left,
        "y": top,
        "width": right - left,
        "height": bottom - top,
        "image": black_mask_clean[top:bottom, left:right].copy(),
    }


def _predict_annotation_cluster(
    cluster_crop: np.ndarray,
    classifier: Optional[AnnotationCandidateClassifier],
    settings: ComponentFilterSettings,
) -> AnnotationPrediction:
    """Use AI when available, otherwise fall back to heuristic cluster rejection."""
    if classifier is None:
        return AnnotationPrediction(
            label="heuristic_annotation_cluster",
            confidence=0.0,
            probabilities={},
        )

    max_dimension = max(cluster_crop.shape[:2])
    if max_dimension > settings.classifier_max_crop_dimension:
        return AnnotationPrediction(
            label="heuristic_annotation_cluster",
            confidence=0.0,
            probabilities={},
        )

    prediction = classifier.predict_crop(cluster_crop)
    if prediction.confidence < settings.classifier_confidence_threshold:
        return AnnotationPrediction(
            label="heuristic_annotation_cluster",
            confidence=float(prediction.confidence),
            probabilities=prediction.probabilities,
        )

    return prediction


def _component_contour_metrics(
    component_mask: np.ndarray,
    bbox_area: float,
) -> dict[str, float]:
    """Measure contour-derived shape statistics for a single component."""
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {
            "contour_area": 0.0,
            "contour_rectangularity": 0.0,
            "perimeter": 0.0,
            "circularity": 0.0,
            "approx_vertices": 0.0,
        }

    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))
    circularity = 0.0
    if perimeter > 0.0:
        circularity = 4.0 * pi * contour_area / (perimeter * perimeter)

    epsilon = max(1.0, 0.04 * perimeter)
    approximated = cv2.approxPolyDP(contour, epsilon, closed=True)

    return {
        "contour_area": contour_area,
        "contour_rectangularity": contour_area / max(1.0, bbox_area),
        "perimeter": perimeter,
        "circularity": circularity,
        "approx_vertices": float(len(approximated)),
    }
