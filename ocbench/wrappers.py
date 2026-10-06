"""Optional Cartesian action spaces for OCBench manipulation environments."""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from ocbench import lie
from ocbench.controllers.ur5e_analytic_ik import AnalyticIKController
from ocbench.envs.manipulation_env import ManipulationEnv


def _validated_action(action, shape):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != shape:
        raise ValueError(f'Expected action shape {shape}, got {action.shape}.')
    if not np.isfinite(action).all():
        raise ValueError('Cartesian actions must be finite.')
    return np.clip(action, -1.0, 1.0)


class CartesianActionWrapper(gym.ActionWrapper):
    """Map (dx, dy, dz, dyaw, gripper) to the CPU environment's joint actions.

    Actions lie in [-1, 1]. Translation is in world axes, yaw is about world z,
    and the gripper points down. Per-step scales come from _ee_action_delta
    (normally 0.05 m, 0.3 rad, and 0.12 gripper units; five times larger in lite
    environments). Positive gripper actions close the fingers, matching the
    underlying environment. Targets are relative to the measured pinch pose,
    clipped to the environment workspace, then converted to attachment poses.

    Joint deltas still obey the underlying environment's action limits, so a
    target may take multiple steps to reach. IK failure holds the arm and keeps
    the gripper command; step info includes the boolean 'ik_no_solution'.
    Observations, rewards, termination, reset, and episode limits are unchanged.
    Call reset before action or step. Released datasets contain joint actions
    and cannot be used as Cartesian actions without conversion.
    """

    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.unwrapped, ManipulationEnv):
            raise TypeError('CartesianActionWrapper requires a CPU OCBench manipulation environment.')
        self.action_space = Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self._ik = AnalyticIKController()
        self.ik_no_solution = False

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.ik_no_solution = False
        return result

    def action(self, action):
        base = self.env.unwrapped
        if base._data is None or base._reset_next_step:
            raise ValueError('Call reset before computing Cartesian actions.')
        delta = _validated_action(action, (5,)) * base._ee_action_delta
        data, model = base._data, base._model
        pos = np.clip(data.site_xpos[base._pinch_site_id] + delta[:3], *base._workspace_bounds)
        rotation = data.site_xmat[base._pinch_site_id].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0]) + delta[3]
        target_pinch = lie.SE3.from_rotation_and_translation(
            lie.SO3.from_z_radians(yaw) @ base._effector_down_rotation, pos
        )
        target_attach = target_pinch @ base._T_pa
        qpos_ids = model.jnt_qposadr[base._arm_joint_ids]
        qpos = data.qpos[qpos_ids]
        self._ik.joint_limits = model.actuator_ctrlrange[base._arm_actuator_ids]
        target, self.ik_no_solution = self._ik.solve_with_status(
            target_attach.translation(), target_attach.rotation().wxyz, qpos
        )
        return base.normalize_action(np.concatenate([target - qpos, delta[4:]])).astype(np.float32)

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(self.action(action))
        return ob, reward, terminated, truncated, dict(info, ik_no_solution=self.ik_no_solution)


class MjWarpCartesianActionWrapper:
    """Batched counterpart of CartesianActionWrapper, with native Warp IK.

    action/step accept (nworld, 5), or broadcast a single (5,) action. The usual
    NumPy step interface copies the resulting joint actions to the host.
    action_gpu and step_cartesian_actions_gpu keep IK and actions on device;
    their input is a float32 Warp array with shape (nworld, 5). Returned device
    buffers are reused on subsequent calls. ik_no_solution_gpu contains one
    integer failure flag per world. GPU inputs must be finite; invalid rows
    produce zero joint/gripper actions and set the failure flag.
    """

    def __init__(self, env):
        from ocbench.mjwarp.envs.manipulation import ManipulationMjWarpEnv

        if not isinstance(env, ManipulationMjWarpEnv):
            raise TypeError('MjWarpCartesianActionWrapper requires an OCBench MJWarp environment.')
        self.env = env
        self.single_action_space = Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, shape=(env.nworld, 5), dtype=np.float32)
        self._joint_actions = None
        self.ik_no_solution_gpu = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _ensure_buffers(self):
        if self.env._data is None:
            raise ValueError('Call reset before computing Cartesian actions.')
        if self._joint_actions is not None:
            return
        from ocbench.mjwarp import analytic_ik

        self._kernels = analytic_ik
        wp = self.env._wp
        device = self.env.data.qpos.device
        cpu = self.env.cpu_env
        self._joint_actions = wp.empty((self.env.nworld, 7), dtype=wp.float32, device=device)
        self.ik_no_solution_gpu = wp.zeros(self.env.nworld, dtype=wp.int32, device=device)
        self._arm_qpos_ids = wp.array(cpu.model.jnt_qposadr[cpu._arm_joint_ids], dtype=wp.int32, device=device)
        self._joint_delta = wp.array(cpu._joint_action_delta, dtype=wp.float32, device=device)
        self._ee_delta = wp.array(cpu._ee_action_delta, dtype=wp.float32, device=device)
        self._limits = wp.array(cpu.model.actuator_ctrlrange[cpu._arm_actuator_ids], dtype=wp.float32, device=device)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Model metadata can change on reset.
        self._joint_actions = None
        self._ensure_buffers()
        return result

    def reset_worlds(self, world_ids, **kwargs):
        result = self.env.reset_worlds(world_ids, **kwargs)
        self._joint_actions = None
        self._ensure_buffers()
        return result

    def action_gpu(self, action):
        """Map device Cartesian actions to a borrowed (nworld, 7) Warp buffer."""
        self._ensure_buffers()
        wp = self.env._wp
        device = self.env.data.qpos.device
        if action.shape != (self.env.nworld, 5) or action.dtype != wp.float32 or action.device != device:
            raise ValueError('Expected float32 Warp actions of shape (nworld, 5) on the simulation device.')
        cpu = self.env.cpu_env
        wp.launch(
            self._kernels.cartesian_actions,
            dim=self.env.nworld,
            inputs=[
                self.env.data.qpos,
                self.env.data.site_xpos,
                self.env.data.site_xmat,
                action,
                self._arm_qpos_ids,
                cpu._pinch_site_id,
                wp.vec3(*cpu._workspace_bounds[0]),
                wp.vec3(*cpu._workspace_bounds[1]),
                wp.vec3(*cpu._T_pa.translation()),
                wp.mat33(cpu._T_pa.rotation().as_matrix()),
                self._ee_delta,
                self._joint_delta,
                self._limits,
                self._joint_actions,
                self.ik_no_solution_gpu,
            ],
            device=device,
        )
        return self._joint_actions

    def action(self, action):
        self._ensure_buffers()
        action = np.asarray(action)
        if action.shape == (5,):
            action = np.broadcast_to(action, (self.env.nworld, 5))
        action = _validated_action(action, (self.env.nworld, 5)).astype(np.float32)
        wp = self.env._wp
        return self.action_gpu(wp.array(action, dtype=wp.float32, device=self.env.data.qpos.device)).numpy()

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(self.action(action))
        return (
            ob,
            reward,
            terminated,
            truncated,
            dict(info, ik_no_solution=self.ik_no_solution_gpu.numpy().astype(bool)),
        )

    def step_cartesian_actions_gpu(self, action, done=None):
        """Map device actions and call the environment's GPU joint-action step."""
        return self.env.step_joint_actions_gpu(self.action_gpu(action), done=done)
