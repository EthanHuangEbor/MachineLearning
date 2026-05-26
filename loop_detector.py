"""
LoopDetector: bag-of-words loop closure detection + geometric verification
Shared by both ORB-SLAM2 and DSO-SLAM systems.
"""

from __future__ import annotations

import cv2
import numpy as np
from slam_base import KeyFrame


# -----------------------------------------------------------------------------
# Visual Vocabulary (simplified k-means based bag-of-words)
# -----------------------------------------------------------------------------

class Vocabulary:
    """Simplified ORB vocabulary using kmeans clustering."""

    def __init__(self, k: int = 10, max_descriptors: int = 50000):
        self.k = k
        self.max_descriptors = max_descriptors
        self.centers = None  # (k, 32) ORB descriptors
        self.idf = np.ones(k)

    def build(self, descriptors_list: list[np.ndarray]) -> None:
        """Build vocabulary from a list of ORB descriptor arrays."""
        all_desc = np.vstack(descriptors_list)
        if len(all_desc) > self.max_descriptors:
            indices = np.random.choice(len(all_desc), self.max_descriptors, replace=False)
            all_desc = all_desc[indices]

        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=self.k, batch_size=1000, random_state=42)
        km.fit(all_desc.astype(np.float32))
        self.centers = km.cluster_centers_.astype(np.uint8)

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        """Convert descriptors to BoW vector (normalized)."""
        if descriptors is None or self.centers is None:
            return np.zeros(self.k)

        # Find nearest center for each descriptor
        # Use simple Euclidean distance on binary descriptors
        bow = np.zeros(self.k, dtype=np.float32)
        for desc in descriptors:
            dists = np.sum((self.centers.astype(np.float32) - desc.astype(np.float32)) ** 2, axis=1)
            idx = np.argmin(dists)
            bow[idx] += 1.0

        # TF-IDF normalization
        if bow.sum() > 0:
            bow = bow / bow.sum()
        bow = bow * np.log(1.0 + 1.0 / (self.idf + 1e-10))
        norm = np.linalg.norm(bow)
        if norm > 0:
            bow /= norm
        return bow

    def get_similarity(self, bow1: np.ndarray, bow2: np.ndarray) -> float:
        """Cosine similarity between two BoW vectors."""
        norm1 = np.linalg.norm(bow1)
        norm2 = np.linalg.norm(bow2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return float(np.dot(bow1, bow2) / (norm1 * norm2))


# -----------------------------------------------------------------------------
# LoopDetector — detect loop closures using BoW + geometric verification
# -----------------------------------------------------------------------------

class LoopDetector:
    def __init__(
        self,
        min_sim: float = 0.08,
        min_matches: int = 20,
        ransac_threshold: float = 3.0,
    ):
        self.min_sim = min_sim
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.vocabulary = Vocabulary(k=10)
        self.candidates: dict[int, list[int]] = {}  # kf_id -> [candidate_kf_ids]
        self.loop_edges: list[tuple[int, int, np.ndarray]] = []  # (kf_i, kf_j, T_ij)

    def add_keyframe(self, kf: KeyFrame, descriptors: np.ndarray | None = None) -> list[int]:
        """Process a new keyframe. Returns loop candidate keyframe IDs."""
        if descriptors is not None and len(descriptors) > 30:
            if self.vocabulary.centers is None:
                history_desc = [
                    old_kf.descriptors
                    for old_kf in getattr(self, "_keyframe_history", [])
                    if old_kf.descriptors is not None and len(old_kf.descriptors) > 30
                ]
                if len(history_desc) >= 3:
                    self.vocabulary.build(history_desc + [descriptors])
            if self.vocabulary.centers is not None:
                kf.bow_vector = self.vocabulary.transform(descriptors)

        candidates = self._find_candidates(kf)
        self._keyframe_history = getattr(self, "_keyframe_history", [])
        self._keyframe_history.append(kf)
        if kf.bow_vector is not None:
            self.bow_history.append((kf.id, kf.bow_vector))
        if candidates:
            self.candidates[kf.id] = candidates
            return candidates
        return []

    def _find_candidates(self, kf: KeyFrame) -> list[int]:
        """Find loop candidates based on BoW similarity."""
        if kf.bow_vector is None:
            return []

        best_sim = 0.0
        best_kf_id = None

        for prev_kf_id, prev_bow in self._history:
            sim = self.vocabulary.get_similarity(kf.bow_vector, prev_bow)
            if sim > best_sim:
                best_sim = sim
                best_kf_id = prev_kf_id

        # Threshold-based selection
        if best_sim < self.min_sim:
            return []
        if best_kf_id is None:
            return []

        # Also add temporal neighbors as candidates (geometric verification will filter)
        return [best_kf_id]

    def _geometric_verification(
        self,
        kf1: KeyFrame,
        kf2: KeyFrame,
        matches: list,
        calib,
    ) -> tuple[bool, np.ndarray]:
        """
        Verify loop closure with RANSAC + essential matrix.
        Returns (is_valid, T_12).
        """
        if len(matches) < self.min_matches:
            return False, np.eye(4)

        # Build pixel correspondences.
        pts1 = []
        pts2 = []
        for m in matches[:self.min_matches]:
            if hasattr(m, 'queryIdx') and hasattr(m, 'trainIdx'):
                if m.queryIdx >= len(kf1.keypoints) or m.trainIdx >= len(kf2.keypoints):
                    continue
                p1 = kf1.keypoints[m.queryIdx].pt
                p2 = kf2.keypoints[m.trainIdx].pt
                pts1.append(p1)
                pts2.append(p2)

        if len(pts1) < 8:
            return False, np.eye(4)

        pts1 = np.array(pts1, dtype=np.float64)
        pts2 = np.array(pts2, dtype=np.float64)

        # Essential matrix from calibrated cameras
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            calib[:3, :3],
            threshold=self.ransac_threshold,
            prob=0.999,
        )

        inliers = mask.ravel() > 0
        if inliers.sum() < self.min_matches * 0.5:
            return False, np.eye(4)

        # Recover pose
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, calib[:3, :3])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        return True, T

    def register_loop(
        self,
        kf1_id: int,
        kf2_id: int,
        T_12: np.ndarray,
    ) -> None:
        """Register a confirmed loop closure edge."""
        self.loop_edges.append((kf1_id, kf2_id, T_12))

    @property
    def _history(self) -> list:
        """Access to past BoW vectors (set by SLAM system)."""
        return self._bow_history

    @_history.setter
    def _history(self, value):
        self._bow_history = value

    @property
    def bow_history(self) -> list:
        return getattr(self, '_bow_history', [])

    @bow_history.setter
    def bow_history(self, val):
        self._bow_history = val


# -----------------------------------------------------------------------------
# Sim3 solver for scale recovery (used in loop correction)
# -----------------------------------------------------------------------------

def solve_sim3(
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Compute Sim(3) transformation (rotation + translation + scale)
    from matched 3D point pairs.

    Returns:
        T: 4x4 similarity transformation
        scale: scaling factor
    """
    centroid1 = pts1.mean(axis=0)
    centroid2 = pts2.mean(axis=0)

    pts1_c = pts1 - centroid1
    pts2_c = pts2 - centroid2

    # Compute scale
    d1 = np.linalg.norm(pts1_c, axis=1)
    d2 = np.linalg.norm(pts2_c, axis=1)
    scale = np.mean(d2) / (np.mean(d1) + 1e-10)

    # SVD for rotation
    H = pts1_c.T @ pts2_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid2 - scale * R @ centroid1

    T = np.eye(4)
    T[:3, :3] = R * scale
    T[:3, 3] = t
    return T, scale
