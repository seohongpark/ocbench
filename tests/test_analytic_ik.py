from pathlib import Path

import mujoco
import numpy as np
import pytest

from ocbench.controllers import ur5e_analytic_ik as ik

MODEL_PATH = Path(__file__).resolve().parents[1] / 'ocbench/descriptions/universal_robots_ur5e/ur5e.xml'


@pytest.fixture(scope='module')
def model():
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def model_pose(model, qpos):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    site = model.site('attachment_site').id
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site])
    return data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy(), quat


def test_fk_and_ik_against_mujoco(model):
    rng = np.random.default_rng(42)
    for _ in range(100):
        qpos = rng.uniform(*model.jnt_range.T)
        pos, rot, quat = model_pose(model, qpos)
        pose = ik.fk_world_attach(qpos)
        np.testing.assert_allclose(pose[:3, 3], pos, atol=1e-12)
        np.testing.assert_allclose(pose[:3, :3], rot, atol=1e-12)
        target, failed = ik.solve_ik_with_status(qpos + 0.01, pos, quat, model.jnt_range)
        assert not failed
        actual_pos, actual_rot, _ = model_pose(model, target)
        np.testing.assert_allclose(actual_pos, pos, atol=1e-7)
        np.testing.assert_allclose(actual_rot, rot, atol=1e-7)
        assert np.all(target >= model.jnt_range[:, 0])
        assert np.all(target <= model.jnt_range[:, 1])
        candidates, valid = ik.inverse_kinematics_dh(ik.world_attach_pose_to_dh(pos, quat))
        assert valid.any()
        for candidate in candidates[valid]:
            actual_pos, actual_rot, _ = model_pose(model, candidate)
            np.testing.assert_allclose(actual_pos, pos, atol=1e-7)
            np.testing.assert_allclose(actual_rot, rot, atol=1e-7)


@pytest.mark.parametrize(
    'q5,q6', [(0, 0.7), (np.pi, -1.1), (1e-8, 0.2), (2e-6, 0.2), (2e-6, 1.5), (np.pi / 2, np.pi / 2)]
)
def test_wrist_singular_and_degenerate_orientations(model, q5, q6):
    qpos = np.array([-1.0, -1.4, 1.2, -0.6, q5, q6])
    pos, rot, quat = model_pose(model, qpos)
    target, failed = ik.solve_ik_with_status(qpos, pos, quat)
    assert not failed
    np.testing.assert_allclose(target, qpos, atol=1e-6)
    actual_pos, actual_rot, _ = model_pose(model, target)
    np.testing.assert_allclose(actual_pos, pos, atol=1e-7)
    np.testing.assert_allclose(actual_rot, rot, atol=1e-7)


@pytest.mark.parametrize('elbow', [0.0, np.pi, 1e-8])
def test_elbow_boundary(model, elbow):
    qpos = np.array([-1.0, -1.4, elbow, -0.6, -1.0, 0.7])
    pos, rot, quat = model_pose(model, qpos)
    target, failed = ik.solve_ik_with_status(qpos, pos, quat)
    assert not failed
    actual_pos, actual_rot, _ = model_pose(model, target)
    np.testing.assert_allclose(actual_pos, pos, atol=1e-7)
    np.testing.assert_allclose(actual_rot, rot, atol=1e-7)


@pytest.mark.parametrize(
    'pos,quat',
    [
        ([10, 0, 0], [1, 0, 0, 0]),
        ([np.nan, 0, 0], [1, 0, 0, 0]),
        ([0, 0, 0], [0, 0, 0, 0]),
        ([0, 0, 0], [np.inf, 0, 0, 0]),
    ],
)
def test_invalid_targets_hold_reference(pos, quat):
    reference = np.arange(6, dtype=float)
    target, failed = ik.solve_ik_with_status(reference, pos, quat)
    assert failed
    np.testing.assert_array_equal(target, reference)


def test_nearest_solution_multiple_turns_and_limits(model):
    qpos = np.array([-1, -1.4, 1.2, -0.6, -1, 0.7])
    pos, _, quat = model_pose(model, qpos)
    reference = qpos + 8 * np.pi
    target, failed = ik.solve_ik_with_status(reference, pos, quat * 3)
    assert not failed
    np.testing.assert_allclose(target, reference, atol=1e-10)
    limits = np.column_stack([qpos - 1e-3, qpos + 1e-3])
    target, failed = ik.solve_ik_with_status(reference, pos, quat, limits)
    assert not failed
    np.testing.assert_allclose(target, qpos, atol=1e-10)
    target, failed = ik.solve_ik_with_status(reference, pos, quat, np.zeros((6, 2)))
    assert failed
    np.testing.assert_array_equal(target, reference)


def test_invalid_reference_and_shapes():
    with pytest.raises(ValueError):
        ik.solve_ik_with_status(np.zeros(7), np.zeros(3), [1, 0, 0, 0])
    with pytest.raises(ValueError):
        ik.solve_ik_with_status(np.full(6, np.nan), np.zeros(3), [1, 0, 0, 0])
