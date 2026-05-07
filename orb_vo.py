from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from kitti_utils import KITTIOdometryLoader, save_trajectory_kitti


@dataclass
class StereoObservation:
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None
    points_3d: np.ndarray
    valid_mask: np.ndarray


class ORBStereoVO:
    def __init__(self, loader: KITTIOdometryLoader, num_features: int = 1500, ratio_test: float = 0.75):
        self.loader = loader
        self.calib = loader.calibration
        self.orb = cv2.ORB_create(nfeatures=num_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio_test = ratio_test
        self.min_inliers = 30

    def extract_stereo_observation(self, left: np.ndarray, right: np.ndarray) -> StereoObservation:
        keypoints_left, descriptors_left = self.orb.detectAndCompute(left, None)
        keypoints_right, descriptors_right = self.orb.detectAndCompute(right, None)

        points_3d = np.zeros((len(keypoints_left), 3), dtype=np.float32)
        valid_mask = np.zeros(len(keypoints_left), dtype=bool)

        if descriptors_left is None or descriptors_right is None or not keypoints_left or not keypoints_right:
            return StereoObservation(keypoints_left, descriptors_left, points_3d, valid_mask)

        matches = self.matcher.knnMatch(descriptors_left, descriptors_right, k=2)
        for candidates in matches:
            if len(candidates) < 2:
                continue
            best, second = candidates
            if best.distance >= self.ratio_test * second.distance:
                continue

            left_pt = keypoints_left[best.queryIdx].pt
            right_pt = keypoints_right[best.trainIdx].pt
            disparity = left_pt[0] - right_pt[0]
            if disparity <= 1.0:
                continue

            z = self.calib.focal_length * self.calib.baseline / disparity
            x = (left_pt[0] - self.calib.cx) * z / self.calib.focal_length
            y = (left_pt[1] - self.calib.cy) * z / self.calib.focal_length
            points_3d[best.queryIdx] = np.array([x, y, z], dtype=np.float32)
            valid_mask[best.queryIdx] = True

        return StereoObservation(keypoints_left, descriptors_left, points_3d, valid_mask)

    def estimate_motion(self, prev_obs: StereoObservation, curr_obs: StereoObservation) -> np.ndarray:
        if prev_obs.descriptors is None or curr_obs.descriptors is None:
            return np.eye(4, dtype=np.float64)

        matches = self.matcher.knnMatch(prev_obs.descriptors, curr_obs.descriptors, k=2)
        object_points = []
        image_points = []

        for candidates in matches:
            if len(candidates) < 2:
                continue
            best, second = candidates
            if best.distance >= self.ratio_test * second.distance:
                continue
            if not prev_obs.valid_mask[best.queryIdx]:
                continue

            object_points.append(prev_obs.points_3d[best.queryIdx])
            image_points.append(curr_obs.keypoints[best.trainIdx].pt)

        if len(object_points) < 6:
            return np.eye(4, dtype=np.float64)

        object_points = np.asarray(object_points, dtype=np.float32)
        image_points = np.asarray(image_points, dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.calib.k_left,
            None,
            reprojectionError=4.0,
            iterationsCount=100,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or inliers is None or len(inliers) < self.min_inliers:
            return np.eye(4, dtype=np.float64)

        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = tvec.reshape(3)
        return transform

    def run(self) -> tuple[list[np.ndarray], list[float]]:
        trajectory = [np.eye(4, dtype=np.float64)]
        runtimes = []

        frames = self.loader.iter_frames()
        prev_frame = next(frames)
        prev_obs = self.extract_stereo_observation(prev_frame.left, prev_frame.right)
        current_pose = np.eye(4, dtype=np.float64)

        for frame in frames:
            start = time.perf_counter()
            curr_obs = self.extract_stereo_observation(frame.left, frame.right)
            relative_pose = self.estimate_motion(prev_obs, curr_obs)
            current_pose = current_pose @ np.linalg.inv(relative_pose)
            trajectory.append(current_pose.copy())
            runtimes.append((time.perf_counter() - start) * 1000.0)
            prev_obs = curr_obs

        return trajectory, runtimes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORB stereo visual odometry on KITTI")
    parser.add_argument("--data-dir", default="data/kitti_odometry", help="KITTI odometry root directory")
    parser.add_argument("--seq", default="00", help="KITTI sequence number")
    parser.add_argument("--output", default=None, help="Output trajectory path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = KITTIOdometryLoader(args.data_dir, args.seq)
    vo = ORBStereoVO(loader)
    trajectory, runtimes = vo.run()

    output_path = args.output or f"results/orb_seq{int(args.seq):02d}.txt"
    save_trajectory_kitti(trajectory, output_path)

    avg_runtime = float(np.mean(runtimes)) if runtimes else 0.0
    print(f"Saved trajectory to {Path(output_path)}")
    print(f"Frames: {len(trajectory)}")
    print(f"Average runtime: {avg_runtime:.2f} ms/frame")


if __name__ == "__main__":
    main()
