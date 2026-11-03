"""Constant-velocity Kalman filter in (cx, cy, a, h) space, as used by SORT/ByteTrack."""
from __future__ import annotations

import numpy as np
import scipy.linalg

# 0.95 quantile of the chi-square distribution (dof 1..9)
CHI2INV95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070,
             6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}


class KalmanFilter:
    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        h = measurement[3]
        std = [2 * self._std_weight_position * h, 2 * self._std_weight_position * h, 1e-2,
               2 * self._std_weight_position * h, 10 * self._std_weight_velocity * h,
               10 * self._std_weight_velocity * h, 1e-5, 10 * self._std_weight_velocity * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        h = mean[3]
        std_pos = [self._std_weight_position * h, self._std_weight_position * h, 1e-2,
                   self._std_weight_position * h]
        std_vel = [self._std_weight_velocity * h, self._std_weight_velocity * h, 1e-5,
                   self._std_weight_velocity * h]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        h = mean[3]
        std = [self._std_weight_position * h, self._std_weight_position * h, 1e-1,
               self._std_weight_position * h]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), (covariance @ self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def mahalanobis(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> float:
        """Squared Mahalanobis distance of one measurement (cx, cy, a, h) to the prediction."""
        projected_mean, projected_cov = self.project(mean, covariance)
        d = measurement - projected_mean
        try:
            sol = np.linalg.solve(projected_cov, d)
        except np.linalg.LinAlgError:  # pragma: no cover
            return float("inf")
        return float(d @ sol)
