"""
ORB-SLAM2 complete system: integrates frontend VO + local mapping + loop closing.
Frontend: feature-point based (ORB detector + PnP + RANSAC)
Backend: local Bundle Adjustment
Loop: BoW-based detection + geometric verification + Pose Graph optimization
"""

from __future__ import annotations

import time
import cv2
import numpy as np
from typing import Optional

from kitti_utils import KITTIOdometryLoader, save_trajectory_kitti
from slam_base import (
    KeyFrame, MapPoint, CovisibilityGraph, EssentialGraph,
    solve_local_ba, solve_pose_graph
)
from local_mapping import LocalMapping, LocalMappingConfig
from loop_detector import LoopDetector
from orb_advanced import GlobalBundleAdjuster, LocalMapTracker, Relocalizer, Sim3LoopFusion


class ORBSLAM2:
    """
    Complete ORB-SLAM2 system with:
    - Tracking: ORB feature extraction, stereo matching, PnP pose estimation
    - Local Mapping: map point creation, local BA
    - Loop Closing: BoW loop detection, Sim3 correction, pose graph optimization
    """

    def __init__(
        self,
        loader: KITTIOdometryLoader,
        num_features: int = 1500,
        min_loop_sim: float = 0.08,
        loop_min_matches: int = 20,
    ):
        self.loader = loader
        self.calib = loader.calibration

        # Frontend components
        self.orb = cv2.ORB_create(nfeatures=num_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Backend components
        self.covisibility = CovisibilityGraph(min_edges=15)
        self.essential_graph = EssentialGraph()
        lm_config = LocalMappingConfig(max_keyframes=8, min_tracked_points=50)
        self.local_mapping = LocalMapping(config=lm_config)
        self.local_mapping.set_covisibility(self.covisibility)
        self.local_mapping.set_calibration(
            self.calib.focal_length, self.calib.focal_length,
            self.calib.cx, self.calib.cy, self.calib.baseline
        )

        # Loop detector
        self.loop_detector = LoopDetector(
            min_sim=min_loop_sim,
            min_matches=loop_min_matches,
        )
        self.loop_detector.bow_history = []  # BoW history for candidate检索

        # State
        self.keyframes: list[KeyFrame] = []
        self.map_points: list[MapPoint] = []
        self.current_kf: Optional[KeyFrame] = None
        self.global_id_counter = 0
        self.trajectory: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
        self.loop_edges: list[tuple[int, int, np.ndarray]] = []
        self.last_inlier_count = 0
        self.frames_since_keyframe = 0
        self.descriptor_history: list[np.ndarray] = []
        self.camera_matrix = self.calib.k_left
        self.local_map_tracker = LocalMapTracker(self.camera_matrix)
        self.relocalizer = Relocalizer(self.camera_matrix)
        self.loop_fusion = Sim3LoopFusion()
        self.global_ba = GlobalBundleAdjuster(self.camera_matrix)
        self.tracking_lost_count = 0
        self.frames_processed = 0
        self.tracking_failures = 0
        self.relocalization_attempts = 0
        self.relocalization_successes = 0
        self.loop_candidates = 0
        self.loop_closures = 0
        self.local_map_refinements = 0
        self.global_ba_runs = 0

    # -------------------------------------------------------------------------
    # Tracking (frontend)
    # -------------------------------------------------------------------------

    def _extract_stereo_observation(self, left: np.ndarray, right: np.ndarray):
        """ORB detection + stereo triangulation."""
        kp1, desc1 = self.orb.detectAndCompute(left, None)
        kp2, desc2 = self.orb.detectAndCompute(right, None)

        points_3d = np.zeros((len(kp1), 3), dtype=np.float32)
        valid_mask = np.zeros(len(kp1), dtype=bool)

        if desc1 is None or desc2 is None:
            return kp1, desc1, points_3d, valid_mask

        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        for candidates in matches:
            if len(candidates) < 2:
                continue
            best, second = candidates
            if best.distance >= 0.75 * second.distance:
                continue

            left_pt = kp1[best.queryIdx].pt
            right_pt = kp2[best.trainIdx].pt
            disparity = left_pt[0] - right_pt[0]
            if disparity <= 1.0:
                continue

            z = self.calib.focal_length * self.calib.baseline / disparity
            x = (left_pt[0] - self.calib.cx) * z / self.calib.focal_length
            y = (left_pt[1] - self.calib.cy) * z / self.calib.focal_length
            points_3d[best.queryIdx] = np.array([x, y, z], dtype=np.float32)
            valid_mask[best.queryIdx] = True

        return kp1, desc1, points_3d, valid_mask

    def _estimate_pose(self, prev_kp, curr_kp, prev_pts_3d, prev_desc, curr_desc):
        """PnP RANSAC pose estimation."""
        self.last_inlier_count = 0
        if prev_desc is None or curr_desc is None:
            return np.eye(4, dtype=np.float64)

        matches = self.matcher.knnMatch(prev_desc, curr_desc, k=2)
        object_pts = []
        image_pts = []

        for candidates in matches:
            if len(candidates) < 2:
                continue
            best, second = candidates
            if best.distance >= 0.75 * second.distance:
                continue
            # Only use valid 3D points
            if best.queryIdx >= len(prev_pts_3d):
                continue
            pt3d = prev_pts_3d[best.queryIdx]
            if np.linalg.norm(pt3d) < 1e-6:
                continue
            object_pts.append(pt3d)
            image_pts.append(curr_kp[best.trainIdx].pt)

        if len(object_pts) < 6:
            return np.eye(4, dtype=np.float64)

        object_pts = np.asarray(object_pts, dtype=np.float32)
        image_pts = np.asarray(image_pts, dtype=np.float32)

        succ, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_pts, image_pts,
            self.calib.k_left, None,
            reprojectionError=4.0,
            iterationsCount=100,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not succ or inliers is None or len(inliers) < 30:
            return np.eye(4, dtype=np.float64)

        self.last_inlier_count = int(len(inliers))
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)
        return T

    def _should_insert_keyframe(self, num_tracked: int) -> bool:
        """Keyframe insertion criterion based on tracked points."""
        return len(self.keyframes) == 0 or num_tracked < 50 or self.frames_since_keyframe >= 3

    # -------------------------------------------------------------------------
    # Local Mapping
    # -------------------------------------------------------------------------

    def _create_keyframe(self, left, right, pose, timestamp) -> KeyFrame:
        kp, desc, pts_3d, valid_mask = self._extract_stereo_observation(left, right)
        kf = KeyFrame(
            id=KeyFrame.next_id(),
            timestamp=timestamp,
            pose=pose,
            left_image=left.copy(),
            right_image=right.copy(),
            keypoints=kp,
            descriptors=desc,
            points_3d=pts_3d,
            valid_mask=valid_mask,
            map_points=[],
        )
        kf.bow_vector = self._compute_bow(desc)
        return kf

    def _compute_bow(self, descriptors) -> Optional[np.ndarray]:
        """Compute bag-of-words vector for a descriptor set."""
        if descriptors is None or len(descriptors) < 30:
            return None

        self.descriptor_history.append(descriptors)
        if len(self.descriptor_history) > 50:
            self.descriptor_history.pop(0)

        vocabulary = self.loop_detector.vocabulary
        if vocabulary.centers is None and len(self.descriptor_history) >= 5:
            vocabulary.build(self.descriptor_history)
        if vocabulary.centers is None:
            return None
        return vocabulary.transform(descriptors)

    # -------------------------------------------------------------------------
    # Loop Closing
    # -------------------------------------------------------------------------

    def _detect_loop(self, kf: KeyFrame) -> Optional[int]:
        """Detect loop closure for a keyframe. Returns loop candidate kf_id or None."""
        if kf.bow_vector is None:
            return None

        best_sim = 0.0
        best_kf_id = None

        # Search in BoW history (skip recent N frames to avoid immediate neighbors)
        history = self.loop_detector.bow_history
        skip_recent = min(10, len(history))

        for i, (hist_kf_id, hist_bow) in enumerate(history[:-skip_recent] if history else []):
            if hist_bow is None:
                continue
            # Compute cosine similarity
            norm1 = np.linalg.norm(kf.bow_vector)
            norm2 = np.linalg.norm(hist_bow)
            if norm1 < 1e-10 or norm2 < 1e-10:
                continue
            sim = np.dot(kf.bow_vector, hist_bow) / (norm1 * norm2)
            if sim > best_sim:
                best_sim = sim
                best_kf_id = hist_kf_id

        if best_sim < self.loop_detector.min_sim:
            return None

        self.loop_candidates += 1
        if best_kf_id is None or not self._verify_loop_candidate(kf, best_kf_id):
            return None

        return best_kf_id

    def _relocalize(self, curr_kp, curr_desc) -> Optional[np.ndarray]:
        self.relocalization_attempts += 1
        if curr_desc is None or not curr_kp:
            return None
        bow = self._compute_bow(curr_desc)
        result = self.relocalizer.relocalize(
            bow,
            curr_kp,
            curr_desc,
            self.keyframes,
            self.loop_detector.vocabulary,
        )
        if result is None:
            return None
        self.last_inlier_count = result.inliers
        self.tracking_lost_count = 0
        self.relocalization_successes += 1
        return result.pose

    def _track_local_map(self, pose_guess: np.ndarray, curr_kp, curr_desc) -> np.ndarray:
        local_points = self.local_map_tracker.collect_local_map(
            self.current_kf,
            self.keyframes,
            self.covisibility,
        )
        result = self.local_map_tracker.refine_pose(pose_guess, curr_kp, curr_desc, local_points)
        if result.inliers > self.last_inlier_count:
            self.last_inlier_count = result.inliers
            self.local_map_refinements += 1
            return result.pose
        return pose_guess

    def _verify_loop_candidate(self, kf: KeyFrame, candidate_id: int) -> bool:
        """Geometric loop verification with ORB matches and essential matrix RANSAC."""
        candidate = next((old_kf for old_kf in self.keyframes if old_kf.id == candidate_id), None)
        if candidate is None or kf.descriptors is None or candidate.descriptors is None:
            return False
        if not kf.keypoints or not candidate.keypoints:
            return False

        raw_matches = self.matcher.knnMatch(kf.descriptors, candidate.descriptors, k=2)
        matches = []
        for candidates in raw_matches:
            if len(candidates) < 2:
                continue
            best, second = candidates
            if best.distance < 0.75 * second.distance:
                matches.append(best)

        if len(matches) < self.loop_detector.min_matches:
            return False

        pts1 = np.float32([kf.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([candidate.keypoints[m.trainIdx].pt for m in matches])
        _, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            self.calib.k_left,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=3.0,
        )
        return mask is not None and int(mask.ravel().sum()) >= self.loop_detector.min_matches

    def _correct_loop(self, loop_kf_id: int, current_kf_id: int) -> None:
        """Correct loop: compute Sim3, update graph, run pose graph optimization."""
        current = next((kf for kf in self.keyframes if kf.id == current_kf_id), None)
        loop = next((kf for kf in self.keyframes if kf.id == loop_kf_id), None)
        if current is None or loop is None:
            return

        sim3 = self.loop_fusion.fuse(current, loop, self.keyframes)
        if sim3 is None:
            return
        self.loop_closures += 1

        # Collect all keyframe poses
        kf_poses = {kf.id: kf.pose for kf in self.keyframes}
        kf_poses[current_kf_id] = self.current_kf.pose if self.current_kf else np.eye(4)

        T_loop_current = np.linalg.inv(kf_poses[loop_kf_id]) @ kf_poses[current_kf_id]
        self.loop_edges.append((loop_kf_id, current_kf_id, T_loop_current))
        self.essential_graph.add_edge(loop_kf_id, current_kf_id, T_loop_current)

        # Run pose graph optimization
        if len(self.loop_edges) > 0:
            residual_count = len(self.loop_edges) * 2 + 6
            variable_count = len(kf_poses) * 6
            if residual_count < variable_count:
                return
            optimized = solve_pose_graph(self.loop_edges, kf_poses, iterations=100)
            # Update keyframe poses
            for kf in self.keyframes:
                if kf.id in optimized:
                    kf.pose = optimized[kf.id]
            if self.global_ba.optimize(self.keyframes):
                self.global_ba_runs += 1

    def get_stats(self) -> dict[str, int]:
        unique_points = {
            mp.id
            for kf in self.keyframes
            for mp in kf.map_points
            if mp.valid
        }
        return {
            "frames_processed": self.frames_processed,
            "keyframes": len(self.keyframes),
            "tracking_failures": self.tracking_failures,
            "relocalization_attempts": self.relocalization_attempts,
            "relocalization_successes": self.relocalization_successes,
            "loop_candidates": self.loop_candidates,
            "loop_closures": self.loop_closures,
            "local_map_refinements": self.local_map_refinements,
            "global_ba_runs": self.global_ba_runs,
            "map_points": len(unique_points),
        }

    # -------------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------------

    def run(self) -> tuple[list[np.ndarray], list[float]]:
        """Run full ORB-SLAM2 system on the loaded sequence."""
        trajectory = [np.eye(4, dtype=np.float64)]
        runtimes = []

        frames = iter(self.loader.iter_frames())
        try:
            first_frame = next(frames)
        except StopIteration:
            return trajectory, runtimes

        current_pose = np.eye(4, dtype=np.float64)
        self.frames_processed = 1
        start = time.perf_counter()
        prev_kp, prev_desc, prev_pts_3d, _ = self._extract_stereo_observation(
            first_frame.left, first_frame.right
        )
        first_kf = self._create_keyframe(
            first_frame.left, first_frame.right, current_pose.copy(), first_frame.timestamp
        )
        self.keyframes.append(first_kf)
        self.covisibility.add_keyframe(first_kf)
        self.current_kf = first_kf
        self.loop_detector.bow_history.append((first_kf.id, first_kf.bow_vector))
        self.local_mapping.insert_keyframe(first_kf)
        self.local_mapping.run()
        runtimes.append((time.perf_counter() - start) * 1000.0)

        for idx, frame in enumerate(frames, start=1):
            start = time.perf_counter()

            # === TRACKING ===
            curr_kp, curr_desc, curr_pts_3d, curr_valid = self._extract_stereo_observation(
                frame.left, frame.right
            )

            # Estimate relative motion if we have previous frame
            relative_pose = self._estimate_pose(
                prev_kp, curr_kp, prev_pts_3d, prev_desc, curr_desc
            )
            if self.last_inlier_count == 0:
                relocalized = self._relocalize(curr_kp, curr_desc)
                if relocalized is not None:
                    current_pose = relocalized
                else:
                    self.tracking_lost_count += 1
                    self.tracking_failures += 1
            else:
                current_pose = current_pose @ np.linalg.inv(relative_pose)
                current_pose = self._track_local_map(current_pose, curr_kp, curr_desc)

            num_tracked = self.last_inlier_count
            self.frames_since_keyframe += 1

            # === KEYFRAME INSERTION ===
            if self._should_insert_keyframe(num_tracked):
                kf = self._create_keyframe(
                    frame.left, frame.right, current_pose.copy(), frame.timestamp
                )
                self.keyframes.append(kf)
                self.covisibility.add_keyframe(kf)
                self.current_kf = kf

                # BoW history
                self.loop_detector.bow_history.append((kf.id, kf.bow_vector))
                if len(self.loop_detector.bow_history) > 100:
                    self.loop_detector.bow_history.pop(0)

                # === LOCAL MAPPING ===
                self.local_mapping.insert_keyframe(kf)
                self.local_mapping.run()
                self.frames_since_keyframe = 0

                # === LOOP CLOSING ===
                loop_candidate_id = self._detect_loop(kf)
                if loop_candidate_id is not None:
                    self._correct_loop(loop_candidate_id, kf.id)

            trajectory.append(current_pose.copy())
            runtimes.append((time.perf_counter() - start) * 1000.0)
            self.frames_processed += 1

            prev_kp, prev_desc, prev_pts_3d = curr_kp, curr_desc, curr_pts_3d

        return trajectory, runtimes


def run_orb_slam(
    data_dir: str,
    seq: str,
    output_path: str | None = None,
    num_features: int = 1500,
) -> tuple[list[np.ndarray], list[float]]:
    loader = KITTIOdometryLoader(data_dir, seq)
    slam = ORBSLAM2(loader, num_features=num_features)
    trajectory, runtimes = slam.run()

    out_path = output_path or f"results/orb_slam_seq{int(seq):02d}.txt"
    save_trajectory_kitti(trajectory, out_path)
    return trajectory, runtimes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ORB-SLAM2 on KITTI")
    parser.add_argument("--data-dir", default="data/kitti_odometry")
    parser.add_argument("--seq", default="00")
    parser.add_argument("--output", default=None)
    parser.add_argument("--features", type=int, default=1500)
    args = parser.parse_args()

    trajectory, runtimes = run_orb_slam(args.data_dir, args.seq, args.output, args.features)
    print(f"ORB-SLAM2: {len(trajectory)} frames, avg {np.mean(runtimes):.2f} ms/frame")
