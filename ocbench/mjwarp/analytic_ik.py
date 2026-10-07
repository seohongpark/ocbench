"""Native Warp version of controllers.ur5e_analytic_ik.

Equations adapted from ur-analytic-ik; see controllers/ur_analytic_ik.LICENSE.
The constants describe OCBench's bundled UR5e MJCF attachment site.
"""

# Explicit scalar constructors make mutable loop variables in Warp.
# ruff: noqa: UP018, RUF046

import warp as wp

from ocbench.mjwarp.primitives.primitive_kernels import _clamp_vec3, _wrap_pi

vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def _dh(theta: float, d: float, a: float, alpha: float) -> wp.mat44:
    c, s = wp.cos(theta), wp.sin(theta)
    ca, sa = wp.cos(alpha), wp.sin(alpha)
    return wp.mat44(
        c,
        -s * ca,
        s * sa,
        a * c,
        s,
        c * ca,
        -c * sa,
        a * s,
        0.0,
        sa,
        ca,
        d,
        0.0,
        0.0,
        0.0,
        1.0,
    )


@wp.func
def fk_dh(q: vec6) -> wp.mat44:
    pose = _dh(q[0], 0.163, 0.0, 1.5707963267948966)
    pose = pose * _dh(q[1], 0.0, -0.425, 0.0)
    pose = pose * _dh(q[2], 0.0, -0.392, 0.0)
    pose = pose * _dh(q[3], 0.134, 0.0, 1.5707963267948966)
    pose = pose * _dh(q[4], 0.1, 0.0, -1.5707963267948966)
    return pose * _dh(q[5], 0.1, 0.0, 0.0)


@wp.func
def solve_ik(pos: wp.vec3, rotation: wp.mat33, reference: vec6, low: vec6, high: vec6):
    """Return (nearest in-limit joint targets, no_solution) for a world pose."""
    world_to_dh = wp.mat33(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    p = world_to_dh * pos
    r = world_to_dh * rotation
    best = reference
    best_distance = float(1.0e30)
    no_solution = int(1)
    a1 = p[0] - 0.1 * r[0, 2]
    b1 = 0.1 * r[1, 2] - p[1]
    domain1 = a1 * a1 + b1 * b1 - 0.134 * 0.134
    if domain1 >= -1.0e-6:
        base1 = wp.atan2(a1, b1)
        delta1 = wp.atan2(wp.sqrt(wp.max(domain1, 0.0)), 0.134)
        for branch in range(8):
            shoulder_sign = float(1.0)
            wrist_sign = float(1.0)
            elbow_sign = float(1.0)
            if branch >= 4:
                shoulder_sign = -1.0
            if branch % 4 >= 2:
                wrist_sign = -1.0
            if branch % 2 == 1:
                elbow_sign = -1.0
            q1 = base1 + shoulder_sign * delta1
            c1, s1 = wp.cos(q1), wp.sin(q1)
            c5 = s1 * r[0, 2] - c1 * r[1, 2]
            h1 = c1 * r[1, 1] - s1 * r[0, 1]
            h2 = s1 * r[0, 0] - c1 * r[1, 0]
            abs_s5 = wp.sqrt(h1 * h1 + h2 * h2)
            q5 = wp.atan2(wrist_sign * abs_s5, c5)
            q6 = wp.atan2(wrist_sign * h1, wrist_sign * h2)
            if abs_s5 <= 1.0e-6:
                q6 = reference[5]
            c5, s5 = wp.cos(q5), wp.sin(q5)
            c6, s6 = wp.cos(q6), wp.sin(q6)
            a234 = c1 * r[0, 0] + s1 * r[1, 0]
            b234 = c1 * r[0, 1] + s1 * r[1, 1]
            h1 = c5 * (c6 * r[2, 0] - s6 * r[2, 1]) - s5 * r[2, 2]
            h2 = c5 * (c6 * a234 - s6 * b234) - s5 * (c1 * r[0, 2] + s1 * r[1, 2])
            q234 = wp.atan2(h1, h2)
            c234, s234 = wp.cos(q234), wp.sin(q234)
            kc = c1 * p[0] + s1 * p[1] - s234 * 0.1 + c234 * s5 * 0.1
            ks = p[2] - 0.163 + c234 * 0.1 + s234 * s5 * 0.1
            c3 = (ks * ks + kc * kc - 0.425 * 0.425 - 0.392 * 0.392) / (2.0 * 0.425 * 0.392)
            domain3 = 1.0 - c3 * c3
            if domain3 >= -1.0e-6:
                q3 = elbow_sign * wp.atan2(wp.sqrt(wp.max(domain3, 0.0)), c3)
                q2 = wp.atan2(ks, kc) - wp.atan2(-0.392 * wp.sin(q3), -0.392 * wp.cos(q3) - 0.425)
                candidate = vec6(q1, q2, q3, q234 - q2 - q3, q5, q6)
                valid = bool(True)
                distance = float(0.0)
                for joint in range(6):
                    angle = reference[joint] + _wrap_pi(candidate[joint] - reference[joint])
                    min_turns = wp.ceil((low[joint] - angle) / 6.283185307179586)
                    max_turns = wp.floor((high[joint] - angle) / 6.283185307179586)
                    angle += 6.283185307179586 * wp.clamp(0.0, min_turns, max_turns)
                    candidate[joint] = angle
                    if not wp.isfinite(angle) or angle < low[joint] or angle > high[joint]:
                        valid = False
                    distance += wp.abs(angle - reference[joint])
                pose = fk_dh(candidate)
                for row in range(3):
                    if not wp.isfinite(p[row]) or wp.abs(pose[row, 3] - p[row]) > 1.0e-3:
                        valid = False
                    for col in range(3):
                        if not wp.isfinite(r[row, col]) or wp.abs(pose[row, col] - r[row, col]) > 1.0e-3:
                            valid = False
                if valid and distance < best_distance:
                    best = candidate
                    best_distance = distance
                    no_solution = 0
    return best, no_solution


@wp.kernel
def cartesian_actions(
    qpos: wp.array2d[float],
    site_xpos: wp.array2d[wp.vec3],
    site_xmat: wp.array2d[wp.mat33],
    actions: wp.array2d[float],
    arm_qpos_ids: wp.array(dtype=int),
    pinch_site_id: int,
    workspace_low: wp.vec3,
    workspace_high: wp.vec3,
    pinch_to_attach_pos: wp.vec3,
    pinch_to_attach_rot: wp.mat33,
    ee_delta: wp.array(dtype=float),
    joint_delta: wp.array(dtype=float),
    limits: wp.array2d[float],
    joint_actions: wp.array2d[float],
    no_solution: wp.array(dtype=int),
):
    world = wp.tid()
    finite = bool(True)
    for i in range(5):
        if not wp.isfinite(actions[world, i]):
            finite = False
    for i in range(7):
        joint_actions[world, i] = 0.0
    no_solution[world] = 1
    if finite:
        delta = wp.vec3(
            wp.clamp(actions[world, 0], -1.0, 1.0) * ee_delta[0],
            wp.clamp(actions[world, 1], -1.0, 1.0) * ee_delta[1],
            wp.clamp(actions[world, 2], -1.0, 1.0) * ee_delta[2],
        )
        pos = _clamp_vec3(site_xpos[world, pinch_site_id] + delta, workspace_low, workspace_high)
        rot = site_xmat[world, pinch_site_id]
        yaw = wp.atan2(rot[1, 0], rot[0, 0]) + wp.clamp(actions[world, 3], -1.0, 1.0) * ee_delta[3]
        c, s = wp.cos(yaw), wp.sin(yaw)
        down = wp.mat33(c, s, 0.0, s, -c, 0.0, 0.0, 0.0, -1.0)
        attach_pos = pos + down * pinch_to_attach_pos
        attach_rot = down * pinch_to_attach_rot
        reference, low, high = vec6(), vec6(), vec6()
        for joint in range(6):
            reference[joint] = qpos[world, arm_qpos_ids[joint]]
            low[joint] = limits[joint, 0]
            high[joint] = limits[joint, 1]
        target, failed = solve_ik(attach_pos, attach_rot, reference, low, high)
        no_solution[world] = failed
        for joint in range(6):
            joint_actions[world, joint] = wp.clamp((target[joint] - reference[joint]) / joint_delta[joint], -1.0, 1.0)
        joint_actions[world, 6] = wp.clamp(actions[world, 4], -1.0, 1.0) * ee_delta[4] / joint_delta[6]
