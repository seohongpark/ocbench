"""Analytic IK for the attachment site of OCBench's bundled UR5e model.

NumPy adaptation of the UR5e analytic equations from ur-analytic-ik (MIT); see
ur_analytic_ik.LICENSE. Dimensions match the MJCF, not factory calibration.
Poses use the model's world frame; quaternions are scalar-first (w, x, y, z).
This solver does not perform collision checking."""

import numpy as np

from ocbench import lie

D1 = 0.163
D4 = 0.134
D5 = 0.1
D6 = 0.1
A2 = -0.425
A3 = -0.392
IK_EPS = 1e-6
IK_FK_TOL = 1e-3
WORLD_TO_DH_ROT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
DH_TO_WORLD_ROT = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return np.mod(angle + np.pi, 2.0 * np.pi) - np.pi


def _safe_sqrt(value: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(value, 0.0))


def _dh_matrix(
    theta: np.ndarray,
    d: float | np.ndarray,
    a: float | np.ndarray,
    alpha: float | np.ndarray,
) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array(
        [
            [c, -s * ca, s * sa, a * c],
            [s, c * ca, -c * sa, a * s],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fk_dh(qpos: np.ndarray) -> np.ndarray:
    """Returns the UR5e DH-frame flange pose for six arm joints."""
    pose = np.eye(4, dtype=np.float64)
    pose = pose @ _dh_matrix(qpos[0], D1, 0.0, np.pi / 2.0)
    pose = pose @ _dh_matrix(qpos[1], 0.0, A2, 0.0)
    pose = pose @ _dh_matrix(qpos[2], 0.0, A3, 0.0)
    pose = pose @ _dh_matrix(qpos[3], D4, 0.0, np.pi / 2.0)
    pose = pose @ _dh_matrix(qpos[4], D5, 0.0, -np.pi / 2.0)
    return pose @ _dh_matrix(qpos[5], D6, 0.0, 0.0)


def fk_world_attach(qpos: np.ndarray) -> np.ndarray:
    """Returns the MuJoCo world-frame attachment-site pose for six arm joints."""
    dh_pose = fk_dh(qpos)
    dh_to_world = np.asarray(DH_TO_WORLD_ROT)
    world_pose = np.eye(4, dtype=np.float64)
    world_pose[:3, :3] = dh_to_world @ dh_pose[:3, :3]
    world_pose[:3, 3] = dh_to_world @ dh_pose[:3, 3]
    return world_pose


def world_attach_pose_to_dh(target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    world_to_dh = np.asarray(WORLD_TO_DH_ROT)
    world_rot = lie.SO3(target_quat / np.linalg.norm(target_quat)).as_matrix()
    dh_pose = np.eye(4, dtype=np.float64)
    dh_pose[:3, :3] = world_to_dh @ world_rot
    dh_pose[:3, 3] = world_to_dh @ target_pos
    return dh_pose


def _calculate_theta6(
    sign5: np.ndarray,
    c1: np.ndarray,
    s1: np.ndarray,
    r11: np.ndarray,
    r12: np.ndarray,
    r21: np.ndarray,
    r22: np.ndarray,
) -> np.ndarray:
    h1 = c1 * r22 - s1 * r12
    h2 = s1 * r11 - c1 * r21
    return np.arctan2(sign5 * h1, sign5 * h2)


def inverse_kinematics_dh(target_pose: np.ndarray, q6_reference: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Returns 8 UR5e IK candidates plus a validity mask.

    Analytic equations adapted from:
    https://github.com/Victorlouisdg/ur-analytic-ik
    https://raw.githubusercontent.com/Victorlouisdg/ur-analytic-ik/main/src/ur_analytic_ik/inverse_kinematics.hh
    """

    r11, r12, r13 = target_pose[0, 0], target_pose[0, 1], target_pose[0, 2]
    r21, r22, r23 = target_pose[1, 0], target_pose[1, 1], target_pose[1, 2]
    r31, r32, r33 = target_pose[2, :3]
    px, py, pz = target_pose[0, 3], target_pose[1, 3], target_pose[2, 3]

    solutions = np.zeros((8, 6), dtype=np.float64)
    valid = np.ones((8,), dtype=np.bool_)

    a1 = px - D6 * r13
    b1 = D6 * r23 - py
    theta1_domain = a1 * a1 + b1 * b1 - D4 * D4
    theta1_valid = theta1_domain >= -IK_EPS
    theta1_delta = np.arctan2(_safe_sqrt(theta1_domain), D4)
    theta1_base = np.arctan2(a1, b1)
    theta1a = theta1_base + theta1_delta
    theta1b = theta1_base - theta1_delta
    solutions[:4, 0] = theta1a
    solutions[4:, 0] = theta1b
    valid = valid & theta1_valid

    for row in (0, 4):
        theta1 = solutions[row, 0]
        c1 = np.cos(theta1)
        s1 = np.sin(theta1)

        c5 = s1 * r13 - c1 * r23
        s5 = _safe_sqrt((s1 * r11 - c1 * r21) ** 2 + (s1 * r12 - c1 * r22) ** 2)
        theta5a = np.arctan2(s5, c5)
        theta5b = np.arctan2(-s5, c5)

        theta6a = _calculate_theta6(1.0, c1, s1, r11, r12, r21, r22)
        theta6b = _calculate_theta6(-1.0, c1, s1, r11, r12, r21, r22)

        # At a wrist singularity theta6 is free; retain the reference angle.
        if s5 <= IK_EPS:
            theta6a = theta6b = q6_reference

        solutions[row : row + 2, 4] = theta5a
        solutions[row : row + 2, 5] = theta6a
        solutions[row + 2 : row + 4, 4] = theta5b
        solutions[row + 2 : row + 4, 5] = theta6b

    for row in (0, 2, 4, 6):
        theta1 = solutions[row, 0]
        theta5 = solutions[row, 4]
        theta6 = solutions[row, 5]

        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        c5 = np.cos(theta5)
        s5 = np.sin(theta5)
        c6 = np.cos(theta6)
        s6 = np.sin(theta6)

        a234 = c1 * r11 + s1 * r21
        b234 = c1 * r12 + s1 * r22
        # Combine both wrist projections to avoid degeneracy at c5 = c6 = 0.
        h1 = c5 * (c6 * r31 - s6 * r32) - s5 * r33
        h2 = c5 * (c6 * a234 - s6 * b234) - s5 * (c1 * r13 + s1 * r23)
        theta234 = np.arctan2(h1, h2)
        c234 = np.cos(theta234)
        s234 = np.sin(theta234)

        kc = c1 * px + s1 * py - s234 * D5 + c234 * s5 * D6
        ks = pz - D1 + c234 * D5 + s234 * s5 * D6
        c3 = (ks * ks + kc * kc - A2 * A2 - A3 * A3) / (2.0 * A2 * A3)
        theta3_domain = 1.0 - c3 * c3
        theta3_valid = theta3_domain >= -IK_EPS
        theta3a = np.arctan2(_safe_sqrt(theta3_domain), c3)
        theta3b = -theta3a

        c3a = np.cos(theta3a)
        s3a = np.sin(theta3a)
        theta2a = np.arctan2(ks, kc) - np.arctan2(A3 * s3a, A3 * c3a + A2)

        c3b = np.cos(theta3b)
        s3b = np.sin(theta3b)
        theta2b = np.arctan2(ks, kc) - np.arctan2(A3 * s3b, A3 * c3b + A2)

        solutions[row, 1:4] = [theta2a, theta3a, theta234 - theta2a - theta3a]
        solutions[row + 1, 1:4] = [theta2b, theta3b, theta234 - theta2b - theta3b]
        valid[row : row + 2] = valid[row : row + 2] & theta3_valid

    solutions = wrap_to_pi(solutions)
    fk_poses = np.stack([fk_dh(q) for q in solutions])
    fk_error = np.max(np.abs(fk_poses - target_pose), axis=(1, 2))
    valid = valid & np.isfinite(solutions).all(axis=1)
    valid = valid & np.isfinite(fk_poses).all(axis=(1, 2))
    valid = valid & (fk_error <= IK_FK_TOL)
    return solutions, valid


def closest_ik_solution(solutions, valid, qpos0, joint_limits=None):
    """Choose the smallest L1 joint displacement, accounting for full turns.

    Optional limits have shape (6, 2). If no candidate is valid, return qpos0.
    """
    candidates = qpos0 + wrap_to_pi(solutions - qpos0)
    valid = np.array(valid, dtype=bool, copy=True)
    if joint_limits is not None:
        limits = np.asarray(joint_limits)
        turns = np.clip(
            np.zeros_like(candidates),
            np.ceil((limits[:, 0] - candidates) / (2 * np.pi)),
            np.floor((limits[:, 1] - candidates) / (2 * np.pi)),
        )
        candidates += 2 * np.pi * turns
        valid &= np.all((candidates >= limits[:, 0]) & (candidates <= limits[:, 1]), axis=1)
    valid &= np.isfinite(candidates).all(axis=1)
    distances = np.where(valid, np.sum(np.abs(candidates - qpos0), axis=1), np.inf)
    if not np.any(valid):
        return np.array(qpos0, copy=True), True
    return candidates[np.argmin(distances)], False


def solve_ik_with_status(qpos0, target_pos, target_quat, joint_limits=None):
    """Return (six joint targets, no_solution), holding qpos0 on failure.

    qpos0, target_pos and target_quat have shapes (6,), (3,), and (4,).
    A zero or nonfinite quaternion and nonfinite target position are invalid.
    The nearest valid branch is selected in joint space, within joint_limits
    when supplied. Targets are flange/attachment poses, not gripper poses.
    """
    qpos0 = np.asarray(qpos0, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    target_quat = np.asarray(target_quat, dtype=float)
    if qpos0.shape != (6,) or target_pos.shape != (3,) or target_quat.shape != (4,):
        raise ValueError('Expected shapes (6,), (3,), and (4,) for joints, position, and quaternion.')
    if not np.isfinite(qpos0).all():
        raise ValueError('Reference joint positions must be finite.')
    if joint_limits is not None:
        joint_limits = np.asarray(joint_limits, dtype=float)
        if (
            joint_limits.shape != (6, 2)
            or not np.isfinite(joint_limits).all()
            or np.any(joint_limits[:, 0] > joint_limits[:, 1])
        ):
            raise ValueError('Joint limits must be finite ordered bounds with shape (6, 2).')
    quat_norm = np.linalg.norm(target_quat)
    if not np.isfinite(target_pos).all() or not np.isfinite(quat_norm) or quat_norm < IK_EPS:
        return qpos0.copy(), True
    target_pose_dh = world_attach_pose_to_dh(target_pos, target_quat)
    with np.errstate(over='ignore', invalid='ignore'):
        solutions, valid = inverse_kinematics_dh(target_pose_dh, qpos0[5])
    return closest_ik_solution(solutions, valid, qpos0, joint_limits)


def solve_ik(qpos0, target_pos, target_quat, joint_limits=None):
    """Return joint targets, or the reference joints when no solution exists."""
    return solve_ik_with_status(qpos0, target_pos, target_quat, joint_limits)[0]


class AnalyticIKController:
    """UR5e analytic alternative to DiffIKController for the attachment site."""

    def __init__(self, joint_limits=None):
        self.joint_limits = joint_limits

    def solve(self, pos, quat, curr_qpos):
        return solve_ik(curr_qpos, pos, quat, self.joint_limits)

    def solve_with_status(self, pos, quat, curr_qpos):
        return solve_ik_with_status(curr_qpos, pos, quat, self.joint_limits)
