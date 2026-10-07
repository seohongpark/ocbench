import mujoco
import numpy as np
import pytest

import ocbench
from ocbench import lie
from ocbench.wrappers import CartesianActionWrapper


@pytest.fixture
def env():
    env = CartesianActionWrapper(ocbench.make('block-cpu-single-task1-v0'))
    env.reset(seed=0)
    yield env
    env.close()


def test_mapping_and_joint_limits(env):
    base = env.unwrapped
    np.testing.assert_allclose(env.action(np.zeros(5)), 0, atol=1e-5)
    action = np.array([0.01, -0.02, 0.01, 0.02, 0.5])
    joint_action = env.action(action)
    assert not env.ik_no_solution
    assert base.action_space.contains(joint_action)
    data = mujoco.MjData(base.model)
    data.qpos[:] = base.data.qpos
    data.qpos[base._arm_joint_ids] += joint_action[:6] * base._joint_action_delta[:6]
    mujoco.mj_forward(base.model, data)
    expected_pos = base.data.site_xpos[base._pinch_site_id] + action[:3] * base._ee_action_delta[:3]
    np.testing.assert_allclose(data.site_xpos[base._pinch_site_id], expected_pos, atol=1e-7)
    initial_rot = lie.SO3.from_matrix(base.data.site_xmat[base._pinch_site_id].reshape(3, 3))
    expected_yaw = initial_rot.compute_yaw_radians() + action[3] * base._ee_action_delta[3]
    expected_rot = (lie.SO3.from_z_radians(expected_yaw) @ base._effector_down_rotation).as_matrix()
    np.testing.assert_allclose(data.site_xmat[base._pinch_site_id].reshape(3, 3), expected_rot, atol=1e-7)
    assert joint_action[6] == pytest.approx(0.5)
    np.testing.assert_allclose(env.action(np.full(5, 3.0)), env.action(np.ones(5)))
    assert np.max(np.abs(env.action(np.ones(5)))) <= 1


def test_failure_holds_arm_and_preserves_gripper(env):
    env.unwrapped._T_pa = lie.SE3.from_rotation_and_translation(lie.SO3.identity(), np.full(3, 10.0))
    action = env.action(np.array([0, 0, 0, 0, 0.5]))
    assert env.ik_no_solution
    np.testing.assert_array_equal(action[:6], np.zeros(6))
    assert action[6] == pytest.approx(0.5)
    assert env.step(np.zeros(5))[-1]['ik_no_solution']


@pytest.mark.parametrize(
    'axis, bound, direction', [(0, 0.10, -1), (0, 0.75, 1), (1, -0.65, -1), (1, 0.65, 1), (2, 0.0, -1), (2, 0.45, 1)]
)
@pytest.mark.parametrize('workspace_scale', [1.0, 0.8])
def test_workspace_clips_pinch_target(env, monkeypatch, axis, bound, direction, workspace_scale):
    base = env.unwrapped
    # Custom environment bounds must also flow through the wrapper.
    base._workspace_bounds *= workspace_scale
    bound *= workspace_scale
    original_bounds = base._workspace_bounds.copy()
    position = np.array([0.4, 0.0, 0.2])
    position[axis] = bound - direction * 0.01
    base.data.site_xpos[base._pinch_site_id] = position
    targets = []

    def capture_target(pos, quat, qpos):
        targets.append(lie.SE3.from_rotation_and_translation(lie.SO3(quat), pos) @ base._T_pa.inverse())
        return qpos.copy(), False

    monkeypatch.setattr(env._ik, 'solve_with_status', capture_target)
    action = np.zeros(5)
    action[axis] = direction
    env.action(action)
    position[axis] = bound
    np.testing.assert_allclose(targets[0].translation(), position, atol=1e-12)
    np.testing.assert_array_equal(base._workspace_bounds, original_bounds)


def test_input_validation_and_reset(env):
    for action in [np.zeros(7), np.zeros((1, 5)), np.full(5, np.nan), np.full(5, np.inf)]:
        with pytest.raises(ValueError):
            env.action(action)
    env.ik_no_solution = True
    env.reset(seed=0)
    assert not env.ik_no_solution


@pytest.mark.parametrize(
    'name',
    [
        'block-cpu-single-task1-v0',
        'block-cpu-lite-single-task1-v0',
        'chamber-cpu-easy-task1-v0',
        'switch-cpu-3x3-task1-v0',
        'hanoi-cpu-single-task1-v0',
        'bowling-cpu-task1-v0',
    ],
)
def test_cpu_families_and_time_limit(name):
    with CartesianActionWrapper(ocbench.make(name, max_episode_steps=2)) as env:
        obs, _ = env.reset(seed=0)
        assert env.action_space.shape == (5,)
        next_obs, _reward, _terminated, truncated, info = env.step(np.zeros(5))
        assert obs.shape == next_obs.shape
        assert np.isfinite(next_obs).all()
        assert not truncated
        assert not info['ik_no_solution']
        assert env.step(np.zeros(5))[3]


def test_uninitialized_cpu_environment():
    from ocbench.envs.block_env import BlockEnv

    with CartesianActionWrapper(BlockEnv()) as env, pytest.raises(ValueError, match='reset'):
        env.action(np.zeros(5))
