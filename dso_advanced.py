from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from kitti_utils import CameraCalibration


class PhotometricCalibrator:
    """Response/vignetting style correction used before direct photometric alignment."""

    def __init__(self, gamma: float = 1.0, vignette_strength: float = 0.0):
        self.gamma = gamma
        self.vignette_strength = vignette_strength
        self._vignette_cache: dict[tuple[int, int], np.ndarray] = {}

    def _vignette(self, shape: tuple[int, int]) -> np.ndarray:
        if shape in self._vignette_cache:
            return self._vignette_cache[shape]
        h, w = shape
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        x = (xs - (w - 1) * 0.5) / max(w, 1)
        y = (ys - (h - 1) * 0.5) / max(h, 1)
        r2 = x * x + y * y
        vignette = 1.0 - self.vignette_strength * r2
        vignette = np.clip(vignette, 0.35, 1.0)
        self._vignette_cache[shape] = vignette.astype(np.float32)
        return self._vignette_cache[shape]

    def correct(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        if abs(self.gamma - 1.0) > 1e-6:
            img = np.power(np.clip(img, 0.0, 1.0), self.gamma)
        img = img / self._vignette(image.shape)
        return np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)


@dataclass
class InverseDepthPoint:
    uv: np.ndarray
    inverse_depth: float
    intensity: float
    weight: float = 1.0
    age: int = 0
    observations: int = 0
    bad_count: int = 0
    last_residual: float = 0.0


@dataclass
class MarginalizationPrior:
    """Lightweight pose prior left by marginalized keyframes."""

    pose_priors: dict[int, np.ndarray] = field(default_factory=dict)
    weight: float = 1e-2

    def add_pose_prior(self, keyframe_id: int, pose: np.ndarray) -> None:
        self.pose_priors[keyframe_id] = pose.copy()

    def residual(self, keyframe_id: int, pose_xi: np.ndarray, reference_xi: np.ndarray) -> np.ndarray:
        if keyframe_id not in self.pose_priors:
            return np.empty(0, dtype=np.float64)
        return (pose_xi[:6] - reference_xi[:6]) * self.weight


class InverseDepthActiveWindow:
    """DSO-style active window storing inverse-depth points per keyframe."""

    def __init__(self, calib: CameraCalibration, max_keyframes: int = 5, max_points_per_kf: int = 600):
        self.calib = calib
        self.max_keyframes = max_keyframes
        self.max_points_per_kf = max_points_per_kf
        self.keyframes: list = []
        self.points_by_kf: dict[int, list[InverseDepthPoint]] = {}
        self.prior = MarginalizationPrior()
        self.total_culled_points = 0

    def reset(self) -> None:
        self.keyframes = []
        self.points_by_kf = {}
        self.prior = MarginalizationPrior(weight=self.prior.weight)

    def add_keyframe(self, kf) -> Optional[object]:
        for points in self.points_by_kf.values():
            for point in points:
                point.age += 1

        self.keyframes.append(kf)
        inv_points = []
        depths = getattr(kf, "active_inv_depths", None)
        if depths is None:
            z = np.asarray(kf.active_points_3d[:, 2], dtype=np.float64)
            depths = np.where(z > 1e-6, 1.0 / z, 0.0)
        for uv, inv_depth, intensity in zip(kf.active_uvs, depths, kf.active_intensities):
            if inv_depth <= 0:
                continue
            inv_points.append(
                InverseDepthPoint(
                    uv=np.asarray(uv, dtype=np.float64),
                    inverse_depth=float(inv_depth),
                    intensity=float(intensity),
                )
            )
            if len(inv_points) >= self.max_points_per_kf:
                break
        self.points_by_kf[kf.id] = inv_points

        marginalized = None
        if len(self.keyframes) > self.max_keyframes:
            marginalized = self.keyframes.pop(0)
            self.prior.add_pose_prior(marginalized.id, marginalized.pose)
            self.points_by_kf.pop(marginalized.id, None)
        return marginalized

    def points_3d_for(self, kf) -> np.ndarray:
        points = self.points_by_kf.get(kf.id, [])
        if not points:
            return kf.active_points_3d
        pts = []
        for point in points:
            z = 1.0 / max(point.inverse_depth, 1e-8)
            x = (point.uv[0] - self.calib.cx) * z / self.calib.focal_length
            y = (point.uv[1] - self.calib.cy) * z / self.calib.focal_length
            pts.append([x, y, z])
        return np.asarray(pts, dtype=np.float64)

    def intensities_for(self, kf) -> np.ndarray:
        points = self.points_by_kf.get(kf.id, [])
        if not points:
            return kf.active_intensities
        return np.asarray([point.intensity for point in points], dtype=np.float64)

    def sample_points(self, kf, max_points: int | None = None) -> list[InverseDepthPoint]:
        points = [point for point in self.points_by_kf.get(kf.id, []) if point.bad_count < 3]
        if max_points is None or len(points) <= max_points:
            return points
        stride = max(1, len(points) // max_points)
        return points[::stride][:max_points]

    def cull_bad_points(
        self,
        *,
        max_bad_count: int = 3,
        max_residual: float = 160.0,
        min_weight: float = 0.05,
    ) -> int:
        removed = 0
        for kf_id, points in list(self.points_by_kf.items()):
            kept = []
            for point in points:
                too_bad = point.bad_count >= max_bad_count
                too_noisy = point.observations > 0 and abs(point.last_residual) > max_residual
                too_weak = point.weight < min_weight
                if too_bad or too_noisy or too_weak:
                    removed += 1
                    continue
                kept.append(point)
            self.points_by_kf[kf_id] = kept
        self.total_culled_points += removed
        return removed

    def refine_inverse_depths(self, ref_kf, target_kf, transform_ref_to_target: np.ndarray, max_step: float = 0.02) -> None:
        """Small finite-difference inverse-depth update for active points."""
        points = self.points_by_kf.get(ref_kf.id, [])
        if not points:
            return
        target = target_kf.gray.astype(np.float32)
        h, w = target.shape
        for point in points:
            inv_depth = max(point.inverse_depth, 1e-8)
            z = 1.0 / inv_depth
            xyz = np.array([
                (point.uv[0] - self.calib.cx) * z / self.calib.focal_length,
                (point.uv[1] - self.calib.cy) * z / self.calib.focal_length,
                z,
            ])
            pt = transform_ref_to_target[:3, :3] @ xyz + transform_ref_to_target[:3, 3]
            if pt[2] <= 0.1:
                point.bad_count += 1
                continue
            u = self.calib.focal_length * pt[0] / pt[2] + self.calib.cx
            v = self.calib.focal_length * pt[1] / pt[2] + self.calib.cy
            if not (1 <= u < w - 2 and 1 <= v < h - 2):
                point.bad_count += 1
                continue
            patch = cv2.getRectSubPix(target, (1, 1), (float(u), float(v)))
            residual = point.intensity - float(patch[0, 0])
            point.last_residual = float(residual)
            point.observations += 1
            if abs(residual) > 120.0:
                point.bad_count += 1
            else:
                point.bad_count = max(0, point.bad_count - 1)
            point.weight = float(np.clip(point.weight * 0.95 + 0.05 * np.exp(-abs(residual) / 50.0), 0.01, 1.0))
            point.inverse_depth = float(np.clip(point.inverse_depth + max_step * np.tanh(residual / 50.0), 1e-4, 2.0))
