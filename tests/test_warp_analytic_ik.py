from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import ocbench
from ocbench import lie
from ocbench.controllers import ur5e_analytic_ik as ik
from ocbench.wrappers import CartesianActionWrapper, MjWarpCartesianActionWrapper

wp = pytest.importorskip('warp')
from ocbench.mjwarp.analytic_ik import solve_ik, vec6


@wp.kernel
def solve_batch(
    positions: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.mat33),
    references: wp.array(dtype=vec6),
    lows: wp.array(dtype=vec6),
    highs: wp.array(dtype=vec6),
    results: wp.array(dtype=vec6),
    failures: wp.array(dtype=int),
):
    i = wp.tid()
    result, failed = solve_ik(positions[i], rotations[i], references[i], lows[i], highs[i])
    results[i] = result
    failures[i] = failed


@pytest.fixture(scope='module', params=['cpu', 'cuda'])
def device(request):
    wp.init()
    if request.param == 'cuda' and not wp.is_cuda_available():
        pytest.skip('CUDA device unavailable')
    return request.param


def test_warp_solver_against_numpy(device):
    rng = np.random.default_rng(42)
    joints = rng.uniform(-2.5, 2.5, (100, 6))
    joints[:10, 4] = 0
    joints[10:20, 4] = np.pi
    joints[20:30, 4:6] = np.pi / 2
    joints[30:40, 2] = 0
    poses = np.stack([ik.fk_world_attach(q) for q in joints])
    poses[-1, :3, 3] = [10, 10, 10]
    references = joints.copy()
    references[40:90] += 0.01
    references[90:95] += 8 * np.pi
    lows = np.full((100, 6), -2 * np.pi)
    highs = -lows
    results = wp.empty(100, dtype=vec6, device=device)
    failures = wp.empty(100, dtype=int, device=device)
    wp.launch(
        solve_batch,
        dim=100,
        inputs=[
            wp.array(poses[:, :3, 3], dtype=wp.vec3, device=device),
            wp.array(poses[:, :3, :3], dtype=wp.mat33, device=device),
            wp.array(references, dtype=vec6, device=device),
            wp.array(lows, dtype=vec6, device=device),
            wp.array(highs, dtype=vec6, device=device),
            results,
            failures,
        ],
        device=device,
    )
    actual, failed = results.numpy(), failures.numpy()
    assert failed[-1] == 1
    np.testing.assert_allclose(actual[-1], references[-1], atol=1e-6)
    for i in range(99):
        assert not failed[i], i
        np.testing.assert_allclose(ik.fk_world_attach(actual[i]), poses[i], atol=1e-3)
        assert np.all(actual[i] >= lows[i]) and np.all(actual[i] <= highs[i])
        expected, no_sol = ik.solve_ik_with_status(
            references[i],
            poses[i, :3, 3],
            lie.SO3.from_matrix(poses[i, :3, :3]).wxyz,
            np.column_stack([lows[i], highs[i]]),
        )
        assert not no_sol
        # At elbow singularities float32 can split one solution into two close branches.
        assert np.sum(abs(actual[i] - references[i])) == pytest.approx(np.sum(abs(expected - references[i])), abs=3e-3)


def test_batched_wrapper_mapping_without_physics(device):
    # Real MuJoCo state and wrapper, with Warp state arrays on CPU or CUDA.
    # This exercises all mapping code even on machines without CUDA physics.
    batched = ocbench.make('block-single-task1-v0', nworld=3)
    cpu = batched.cpu_env
    cpu.reset(seed=0)
    cpu_wrapper = CartesianActionWrapper(cpu)
    states, positions, rotations = [], [], []
    for seed in range(3):
        cpu.reset(seed=seed)
        states.append(cpu.data.qpos.copy())
        positions.append(cpu.data.site_xpos.copy())
        rotations.append(cpu.data.site_xmat.reshape(-1, 3, 3).copy())
    batched._wp = wp
    batched._data = SimpleNamespace(
        qpos=wp.array(np.array(states), dtype=float, device=device),
        site_xpos=wp.array(np.array(positions), dtype=wp.vec3, device=device),
        site_xmat=wp.array(np.array(rotations), dtype=wp.mat33, device=device),
    )
    wrapper = MjWarpCartesianActionWrapper(batched)
    actions = np.array([[0, 0, 0, 0, 0], [0.01, 0.02, -0.01, 0.03, 0.5], [1, -1, 1, -1, -1]])
    actual = wrapper.action(actions)
    for i in range(3):
        cpu.data.qpos[:] = states[i]
        mujoco.mj_forward(cpu.model, cpu.data)
        np.testing.assert_allclose(actual[i], cpu_wrapper.action(actions[i]), atol=3e-5)
    # Exercise every workspace face on both backends using the same measured poses.
    for position, direction in [([0.26, -0.34, 0.01], -1), ([0.59, 0.34, 0.34], 1)]:
        clipped_positions = np.array(positions)
        clipped_positions[:, cpu._pinch_site_id] = position
        batched._data.site_xpos = wp.array(clipped_positions, dtype=wp.vec3, device=device)
        boundary_action = np.array([direction, direction, direction, 0, 0])
        actual = wrapper.action(boundary_action)
        for i in range(3):
            cpu.data.qpos[:] = states[i]
            mujoco.mj_forward(cpu.model, cpu.data)
            cpu.data.site_xpos[cpu._pinch_site_id] = position
            np.testing.assert_allclose(actual[i], cpu_wrapper.action(boundary_action), atol=3e-5)
    assert wrapper.action_space.shape == (3, 5)
    assert wrapper.single_action_space.shape == (5,)
    assert wrapper.action(np.zeros(5)).shape == (3, 7)
    for invalid in [np.zeros(7), np.full((3, 5), np.nan)]:
        with pytest.raises(ValueError):
            wrapper.action(invalid)
    actions[1, 0] = np.nan
    output = wrapper.action_gpu(wp.array(actions, dtype=float, device=device)).numpy()
    np.testing.assert_array_equal(output[1], np.zeros(7))
    assert wrapper.ik_no_solution_gpu.numpy()[1] == 1
    cpu.close()


def test_mjwarp_physics_integration():
    pytest.importorskip('mujoco_warp')
    if not wp.is_cuda_available():
        pytest.skip('MJWarp physics requires CUDA')
    wrapper = MjWarpCartesianActionWrapper(ocbench.make('block-single-task1-v0', nworld=2, use_cuda_graph=False))
    try:
        obs, _ = wrapper.reset(seed=0)
        next_obs, _, _, _, info = wrapper.step(np.zeros((2, 5)))
        assert obs.shape == next_obs.shape
        assert np.isfinite(next_obs).all()
        assert not info['ik_no_solution'].any()
        wrapper.reset_worlds([1], seeds=np.array([42], dtype=np.uint32))
        action = wp.zeros((2, 5), dtype=float, device=wrapper.data.qpos.device)
        result = wrapper.step_cartesian_actions_gpu(action)
        assert result.shape == (2,)
    finally:
        wrapper.close()
