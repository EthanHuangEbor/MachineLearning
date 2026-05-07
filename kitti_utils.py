from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

try:
    import pykitti
except ImportError:  # pragma: no cover
    pykitti = None


@dataclass
class StereoFrame:
    index: int
    left: np.ndarray
    right: np.ndarray
    timestamp: Optional[float]


@dataclass
class CameraCalibration:
    k_left: np.ndarray
    k_right: np.ndarray
    p_left: np.ndarray
    p_right: np.ndarray
    focal_length: float
    cx: float
    cy: float
    baseline: float


class KITTIOdometryLoader:
    def __init__(self, base_dir: str | Path, sequence: str):
        self.base_dir = Path(base_dir)
        self.sequence = f"{int(sequence):02d}"
        self.sequence_dir = self.base_dir / "sequences" / self.sequence
        self.poses_path = self.base_dir / "poses" / f"{self.sequence}.txt"

        if pykitti is not None:
            self.dataset = pykitti.odometry(str(self.base_dir), self.sequence)
        else:
            self.dataset = None

        self.calibration = self._load_calibration()
        self.timestamps = self._load_timestamps()
        self.gt_poses = self._load_ground_truth()

    def _load_calibration(self) -> CameraCalibration:
        calib_path = self.sequence_dir / "calib.txt"
        if not calib_path.exists():
            raise FileNotFoundError(f"Missing calibration file: {calib_path}")

        projections: dict[str, np.ndarray] = {}
        with calib_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                key, values = line.split(":", 1)
                projections[key.strip()] = np.fromstring(values, sep=" ").reshape(3, 4)

        p_left = projections["P0"]
        p_right = projections["P1"]
        k_left = p_left[:, :3]
        k_right = p_right[:, :3]
        focal_length = float(k_left[0, 0])
        cx = float(k_left[0, 2])
        cy = float(k_left[1, 2])
        tx_left = float(p_left[0, 3] / focal_length)
        tx_right = float(p_right[0, 3] / focal_length)
        baseline = abs(tx_left - tx_right)

        return CameraCalibration(
            k_left=k_left,
            k_right=k_right,
            p_left=p_left,
            p_right=p_right,
            focal_length=focal_length,
            cx=cx,
            cy=cy,
            baseline=baseline,
        )

    def _load_timestamps(self) -> list[Optional[float]]:
        times_path = self.sequence_dir / "times.txt"
        if not times_path.exists():
            return []
        with times_path.open("r", encoding="utf-8") as f:
            return [float(line.strip()) for line in f if line.strip()]

    def _load_ground_truth(self) -> Optional[np.ndarray]:
        if not self.poses_path.exists():
            return None
        poses = []
        with self.poses_path.open("r", encoding="utf-8") as f:
            for line in f:
                values = np.fromstring(line, sep=" ")
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :4] = values.reshape(3, 4)
                poses.append(pose)
        return np.stack(poses) if poses else None

    def __len__(self) -> int:
        left_dir = self.sequence_dir / "image_0"
        return len(list(left_dir.glob("*.png")))

    def read_image_pair(self, index: int) -> StereoFrame:
        left_path = self.sequence_dir / "image_0" / f"{index:06d}.png"
        right_path = self.sequence_dir / "image_1" / f"{index:06d}.png"
        left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            raise FileNotFoundError(f"Missing stereo pair at index {index:06d}")

        timestamp = self.timestamps[index] if index < len(self.timestamps) else None
        return StereoFrame(index=index, left=left, right=right, timestamp=timestamp)

    def iter_frames(self) -> Iterator[StereoFrame]:
        for index in range(len(self)):
            yield self.read_image_pair(index)


def disparity_to_depth(disparity: np.ndarray, calib: CameraCalibration, min_disparity: float = 0.1) -> np.ndarray:
    disparity = disparity.astype(np.float32)
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > min_disparity
    depth[valid] = calib.focal_length * calib.baseline / disparity[valid]
    return depth


def save_trajectory_kitti(poses: list[np.ndarray], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pose in poses:
            flattened = pose[:3, :].reshape(-1)
            f.write(" ".join(f"{value:.9f}" for value in flattened))
            f.write("\n")


def load_ground_truth_trajectory(base_dir: str | Path, sequence: str) -> Optional[np.ndarray]:
    loader = KITTIOdometryLoader(base_dir, sequence)
    return loader.gt_poses
