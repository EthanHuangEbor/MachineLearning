from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from kitti_utils import CameraCalibration, KITTIOdometryLoader, disparity_to_depth, save_trajectory_kitti


@dataclass
class Keyframe:
    gray: np.ndarray
    depth: np.ndarray
    active_uvs: np.ndarray
    active_points_3d: np.ndarray
    pose: np.ndarray


def build_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [image]
    for _ in range(levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def select_high_gradient_pixels(gray: np.ndarray, depth: np.ndarray, num_points: int = 1500) -> np.ndarray:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    border = 10
    gradient_magnitude[:border, :] = 0
    gradient_magnitude[-border:, :] = 0
    gradient_magnitude[:, :border] = 0
    gradient_magnitude[:, -border:] = 0

    valid_depth = (depth > 0.5) & (depth < 100.0)
    gradient_magnitude[~valid_depth] = 0

    flat_indices = np.argsort(gradient_magnitude.ravel())[::-1][:num_points]
    rows, cols = np.unravel_index(flat_indices, gray.shape)
    return np.column_stack([cols, rows]).astype(np.float32)


def compute_stereo_depth(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=7,
        P1=8 * 3 * 7 ** 2,
        P2=32 * 3 * 7 ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = stereo.compute(left, right).astype(np.float32) / 16.0
    return disparity


def bilinear_interpolate(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = image.shape
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = u0 + 1
    v1 = v0 + 1

    u0 = np.clip(u0, 0, w - 2)
    u1 = np.clip(u1, 0, w - 1)
    v0 = np.clip(v0, 0, h - 2)
    v1 = np.clip(v1, 0, h - 1)

    du = u - u0
    dv = v - v0

    img_float = image.astype(np.float32)
    interpolated = (
        img_float[v0, u0] * (1 - du) * (1 - dv)
        + img_float[v0, u1] * du * (1 - dv)
        + img_float[v1, u0] * (1 - du) * dv
        + img_float[v1, u1] * du * dv
    )
    return interpolated


def pose_to_matrix(xi: np.ndarray) -> np.ndarray:
    tx, ty, tz, rx, ry, rz = xi
    angle = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    if angle < 1e-8:
        rotation = np.eye(3)
    else:
        axis = np.array([rx, ry, rz]) / angle
        c, s = np.cos(angle), np.sin(angle)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * K

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [tx, ty, tz]
    return transform


def compute_photometric_residuals(
    xi: np.ndarray,
    points_3d: np.ndarray,
    intensities_ref: np.ndarray,
    curr_gray_pyramid: list[np.ndarray],
    calib: CameraCalibration,
    level: int,
) -> np.ndarray:
    scale = 0.5 ** level
    scaled_f = calib.focal_length * scale
    scaled_cx = calib.cx * scale
    scaled_cy = calib.cy * scale

    transform = pose_to_matrix(xi)
    pts_cam = (transform[:3, :3] @ points_3d.T).T + transform[:3, 3]

    depth = pts_cam[:, 2]
    valid = depth > 0.1
    u_proj = np.where(valid, scaled_f * pts_cam[:, 0] / depth + scaled_cx, 0.0)
    v_proj = np.where(valid, scaled_f * pts_cam[:, 1] / depth + scaled_cy, 0.0)

    curr_gray = curr_gray_pyramid[level].astype(np.float32)
    h, w = curr_gray.shape
    in_bounds = (u_proj > 0) & (u_proj < w - 1) & (v_proj > 0) & (v_proj < h - 1)
    mask = valid & in_bounds

    residuals = np.zeros(len(points_3d), dtype=np.float32)
    if mask.any():
        interpolated = bilinear_interpolate(curr_gray, u_proj[mask], v_proj[mask])
        residuals[mask] = intensities_ref[mask] - interpolated
        residuals[~mask] = 0.0

    return residuals


class DirectSparseVO:
    def __init__(
        self,
        loader: KITTIOdometryLoader,
        num_active_points: int = 1500,
        pyramid_levels: int = 4,
        keyframe_flow_threshold: float = 20.0,
    ):
        self.loader = loader
        self.calib = loader.calibration
        self.num_active_points = num_active_points
        self.pyramid_levels = pyramid_levels
        self.keyframe_flow_threshold = keyframe_flow_threshold

    def _make_keyframe(self, frame_left: np.ndarray, frame_right: np.ndarray, pose: np.ndarray) -> Keyframe:
        disparity = compute_stereo_depth(frame_left, frame_right)
        depth = disparity_to_depth(disparity, self.calib)
        active_uvs = select_high_gradient_pixels(frame_left, depth, self.num_active_points)

        u = active_uvs[:, 0].astype(np.float32)
        v = active_uvs[:, 1].astype(np.float32)
        z = bilinear_interpolate(depth, u, v)
        valid_depth = z > 0.5
        active_uvs = active_uvs[valid_depth]
        z = z[valid_depth]

        x = (active_uvs[:, 0] - self.calib.cx) * z / self.calib.focal_length
        y = (active_uvs[:, 1] - self.calib.cy) * z / self.calib.focal_length
        points_3d = np.column_stack([x, y, z])

        return Keyframe(gray=frame_left, depth=depth, active_uvs=active_uvs, active_points_3d=points_3d, pose=pose)

    def _estimate_relative_pose(self, keyframe: Keyframe, curr_gray: np.ndarray) -> np.ndarray:
        curr_pyramid = build_pyramid(curr_gray, self.pyramid_levels)
        ref_gray_pyramid = build_pyramid(keyframe.gray, self.pyramid_levels)

        intensities_ref = bilinear_interpolate(
            keyframe.gray.astype(np.float32),
            keyframe.active_uvs[:, 0],
            keyframe.active_uvs[:, 1],
        )
        points_3d = keyframe.active_points_3d

        xi = np.zeros(6, dtype=np.float64)

        for level in range(self.pyramid_levels - 1, -1, -1):
            scale = 0.5 ** level
            ref_intensities_at_level = bilinear_interpolate(
                ref_gray_pyramid[level].astype(np.float32),
                keyframe.active_uvs[:, 0] * scale,
                keyframe.active_uvs[:, 1] * scale,
            )

            def residuals_fn(xi_opt: np.ndarray) -> np.ndarray:
                return compute_photometric_residuals(xi_opt, points_3d, ref_intensities_at_level, curr_pyramid, self.calib, level)

            result = least_squares(residuals_fn, xi, method="lm", max_nfev=20, ftol=1e-4)
            xi = result.x

        return pose_to_matrix(xi)

    def _should_update_keyframe(self, keyframe: Keyframe, curr_gray: np.ndarray) -> bool:
        kps_left, desc_left = cv2.ORB_create(nfeatures=200).detectAndCompute(keyframe.gray, None)
        kps_curr, desc_curr = cv2.ORB_create(nfeatures=200).detectAndCompute(curr_gray, None)
        if desc_left is None or desc_curr is None or not kps_left or not kps_curr:
            return True

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.match(desc_left, desc_curr)
        if not matches:
            return True

        flows = []
        for m in matches[:50]:
            dx = kps_left[m.queryIdx].pt[0] - kps_curr[m.trainIdx].pt[0]
            dy = kps_left[m.queryIdx].pt[1] - kps_curr[m.trainIdx].pt[1]
            flows.append(np.sqrt(dx ** 2 + dy ** 2))

        return float(np.mean(flows)) > self.keyframe_flow_threshold

    def run(self) -> tuple[list[np.ndarray], list[float]]:
        trajectory = [np.eye(4, dtype=np.float64)]
        runtimes = []

        frames = self.loader.iter_frames()
        first_frame = next(frames)
        current_pose = np.eye(4, dtype=np.float64)
        keyframe = self._make_keyframe(first_frame.left, first_frame.right, current_pose)

        for frame in frames:
            start = time.perf_counter()

            relative_pose = self._estimate_relative_pose(keyframe, frame.left)
            current_pose = keyframe.pose @ relative_pose
            trajectory.append(current_pose.copy())
            runtimes.append((time.perf_counter() - start) * 1000.0)

            if self._should_update_keyframe(keyframe, frame.left):
                keyframe = self._make_keyframe(frame.left, frame.right, current_pose)

        return trajectory, runtimes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DSO-style direct sparse visual odometry on KITTI")
    parser.add_argument("--data-dir", default="data/kitti_odometry", help="KITTI odometry root directory")
    parser.add_argument("--seq", default="00", help="KITTI sequence number")
    parser.add_argument("--output", default=None, help="Output trajectory path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np

    loader = KITTIOdometryLoader(args.data_dir, args.seq)
    vo = DirectSparseVO(loader)
    trajectory, runtimes = vo.run()

    output_path = args.output or f"results/direct_seq{int(args.seq):02d}.txt"
    save_trajectory_kitti(trajectory, output_path)

    avg_runtime = float(np.mean(runtimes)) if runtimes else 0.0
    print(f"Saved trajectory to {output_path}")
    print(f"Frames: {len(trajectory)}")
    print(f"Average runtime: {avg_runtime:.2f} ms/frame")


if __name__ == "__main__":
    main()
