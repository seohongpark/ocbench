import itertools

import mujoco
import numpy as np
from dm_control import mjcf

from ocbench import lie
from ocbench.envs.manipulation_env import ManipulationEnv


class BlockEnv(ManipulationEnv):
    def __init__(
        self,
        env_type='single',
        task_id=1,
        cube_colors=None,
        goal_marker_size=None,
        goal_marker_line_width=0.01,
        success_hold_time=1.0,
        **kwargs,
    ):
        """Block environment.

        Args:
            env_type: Number of cubes ('single', 'double', 'triple' or 'quadruple').
            task_id: Task index (1-based).
            cube_colors: Optional color names, one per cube, overriding the default
                ('blue', 'red', 'orange', 'green'). Each name must have a 'light' variant in
                `ManipulationEnv._colors`, used to mark success. CPU environments only.
            goal_marker_size: Optional inner side length (m) of a flat dark-gray square drawn on
                the table around the fixed goal of the 'single' task. When set, success also
                becomes stricter: the whole cube lies inside the square, rests flat on the table,
                touches nothing but the table (released), and is still, for `success_hold_time`
                seconds; cubes never start on or near the square. Default None keeps the stock
                scene and success. CPU environments only.
            goal_marker_line_width: Line width (m) of the goal square.
            success_hold_time: Seconds the strict success conditions must hold continuously.
        """
        self._env_type = env_type
        self._goal_marker_size = goal_marker_size
        self._goal_marker_line_width = goal_marker_line_width
        self._success_hold_time = success_hold_time
        self._goal_marker_xy = None
        self._success_hold_steps = 0
        if goal_marker_size is not None and env_type != 'single':
            raise ValueError('goal_marker_size is only supported for the single-cube task.')
        self._cube_size = 0.03
        self._min_object_init_dist = 0.09

        if self._env_type == 'single':
            self._num_cubes = 1
        elif self._env_type == 'double':
            self._num_cubes = 2
        elif self._env_type == 'triple':
            self._num_cubes = 3
        elif self._env_type == 'quadruple':
            self._num_cubes = 4
        else:
            raise ValueError(f'Invalid env_type: {env_type}')

        super().__init__(
            task_id=task_id,
            **kwargs,
        )

        # Define constants.
        if cube_colors is None:
            cube_colors = ('blue', 'red', 'orange', 'green')
        cube_colors = tuple(cube_colors)
        if len(cube_colors) < self._num_cubes:
            raise ValueError(f'Need {self._num_cubes} cube colors, got {len(cube_colors)}: {cube_colors}')
        for name in cube_colors:
            if name not in self._colors or f'light{name}' not in self._colors:
                raise ValueError(f'Unsupported cube color: {name!r}')
        self._cube_colors = np.array([self._colors[name] for name in cube_colors])
        self._cube_success_colors = np.array([self._colors[f'light{name}'] for name in cube_colors])

        self._arm_sampling_bounds = np.asarray([[0.18, -0.50, 0.20], [0.67, 0.50, 0.35]])
        self._object_sampling_bounds = np.asarray([[0.20, -0.55], [0.62, 0.55]])
        self._target_sampling_bounds = self._object_sampling_bounds

    def set_tasks(self):
        s = self._cube_size
        if self._env_type == 'single':
            self.task_infos = [
                dict(
                    task_name='move',
                    goal_xyzs=np.array([[0.425, 0.0, s]]),
                )
            ]
        elif self._env_type == 'double':
            self.task_infos = [
                dict(
                    task_name='double_pnp',
                    goal_xyzs=np.array(
                        [
                            [0.425, -0.2, s],
                            [0.425, 0.2, s],
                        ]
                    ),
                ),
                dict(
                    task_name='stack_anywhere',
                    success_type='stack_anywhere',
                    hide_targets=True,
                ),
            ]
        elif self._env_type == 'triple':
            self.task_infos = [
                dict(
                    task_name='triple_pnp',
                    goal_xyzs=np.array(
                        [
                            [0.425, 0.0, s],
                            [0.425, 0.2, s],
                            [0.425, -0.2, s],
                        ]
                    ),
                ),
                dict(
                    task_name='stack_anywhere',
                    success_type='stack_anywhere',
                    hide_targets=True,
                ),
            ]
        elif self._env_type == 'quadruple':
            self.task_infos = [
                dict(
                    task_name='quadruple_pnp',
                    goal_xyzs=np.array(
                        [
                            [0.525, -0.2, s],
                            [0.325, 0.2, s],
                            [0.325, -0.2, s],
                            [0.525, 0.2, s],
                        ]
                    ),
                ),
                dict(
                    task_name='stack_anywhere',
                    success_type='stack_anywhere',
                    hide_targets=True,
                ),
                dict(
                    task_name='grid',
                    goal_xyzs=np.array(
                        [
                            [0.425, -s, s],
                            [0.425, s, s],
                            [0.425, -s, 3 * s],
                            [0.425, s, 3 * s],
                        ]
                    ),
                ),
            ]

    def _is_stack_anywhere_task(self):
        return self.cur_task_info is not None and self.cur_task_info.get('success_type') == 'stack_anywhere'

    def _randomize_blocks(self):
        object_xys = []
        for i in range(self._num_cubes):
            for _ in range(100):
                xy = self.np_random.uniform(*self._object_sampling_bounds)
                if all(np.linalg.norm(xy - other_xy) >= self._min_object_init_dist for other_xy in object_xys):
                    break
            object_xys.append(xy)
            obj_pos = np.array((*xy, self._cube_size))
            yaw = self.np_random.uniform(0, 2 * np.pi)
            obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
            self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori

    def add_objects(self, arena_mjcf):
        # Add cube scene.
        cube_outer_mjcf = mjcf.from_path((self._desc_dir / 'cube_outer.xml').as_posix())
        cube_outer_mjcf.find('default', 'cube').geom.size = np.full(3, self._cube_size)
        arena_mjcf.include_copy(cube_outer_mjcf)

        # Add `num_cubes` cubes to the scene.
        distance = self._cube_size * 2 + 0.01
        for i in range(self._num_cubes):
            cube_mjcf = mjcf.from_path((self._desc_dir / 'cube_inner.xml').as_posix())
            cube_mjcf.find('body', 'object_0').pos[2] = self._cube_size
            cube_mjcf.find('body', 'object_target_0').pos[2] = self._cube_size
            pos = -distance * (self._num_cubes - 1) + 2 * distance * i
            cube_mjcf.find('body', 'object_0').pos[1] = pos
            cube_mjcf.find('body', 'object_target_0').pos[1] = pos
            for tag in ['body', 'joint', 'geom', 'site']:
                for item in cube_mjcf.find_all(tag):
                    if hasattr(item, 'name') and item.name is not None and item.name.endswith('_0'):
                        item.name = item.name[:-2] + f'_{i}'
            arena_mjcf.include_copy(cube_mjcf)

        # Save cube geoms.
        self._cube_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_geoms_list.append(arena_mjcf.find('body', f'object_{i}').find_all('geom'))
        self._cube_target_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_target_geoms_list.append(arena_mjcf.find('body', f'object_target_{i}').find_all('geom'))

        if self._goal_marker_size is not None:
            self._add_goal_marker(arena_mjcf)

        # Add cameras.
        cameras = {
            'front': {
                'pos': (1.945, 0.000, 0.748),
                'xyaxes': (0.000, 1.000, 0.000, -0.342, 0.000, 0.940),
            },
            'front_pixels': {
                'pos': (1.053, -0.014, 0.639),
                'xyaxes': (0.000, 1.000, 0.000, -0.628, 0.001, 0.778),
            },
            'side': {
                'pos': (1.123, -1.424, 0.748),
                'xyaxes': (0.866, 0.500, -0.000, -0.171, 0.296, 0.940),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

    def _add_goal_marker(self, arena_mjcf):
        """Add a flat, collision-free square outline around the fixed goal."""
        goal_xyzs = [info['goal_xyzs'] for info in self.task_infos if 'goal_xyzs' in info]
        if len(goal_xyzs) != 1:
            raise ValueError('The goal marker needs exactly one fixed goal.')
        self._goal_marker_xy = np.asarray(goal_xyzs[0][0][:2], dtype=float)
        gx, gy = self._goal_marker_xy
        line = self._goal_marker_line_width
        half_thickness = 0.0015
        offset = self._goal_marker_size / 2 + line / 2  # Line centre distance from the goal.
        half_length = offset + line / 2  # Lines overlap at the corners.
        sides = {
            'px': ((gx + offset, gy), (line / 2, half_length)),
            'nx': ((gx - offset, gy), (line / 2, half_length)),
            'py': ((gx, gy + offset), (half_length, line / 2)),
            'ny': ((gx, gy - offset), (half_length, line / 2)),
        }
        for name, ((x, y), (hx, hy)) in sides.items():
            arena_mjcf.worldbody.add(
                'geom',
                name=f'goal_marker_{name}',
                type='box',
                pos=(x, y, half_thickness),
                size=(hx, hy, half_thickness),
                rgba=self._colors['darkgray'],
                contype=0,
                conaffinity=0,
                group=1,
            )

    def _near_goal_marker(self):
        """Whether any cube could overlap the goal marker (used to resample initial states)."""
        if self._goal_marker_size is None:
            return False
        clearance = self._goal_marker_size / 2 + self._goal_marker_line_width + self._cube_size * np.sqrt(2)
        for i in range(self._num_cubes):
            xy = self._data.joint(f'object_joint_{i}').qpos[:2]
            if np.max(np.abs(xy - self._goal_marker_xy)) <= clearance:
                return True
        return False

    def goal_marker_conditions(self, i=0):
        """Instantaneous strict-success conditions for cube `i` (goal marker mode)."""
        s = self._cube_size
        gid = self._cube_geom_ids_list[i][0]
        center = self._data.geom_xpos[gid]
        rot = self._data.geom_xmat[gid].reshape(3, 3)
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
        corners = center + (signs * s) @ rot.T
        inside = bool(np.all(np.abs(corners[:, :2] - self._goal_marker_xy) <= self._goal_marker_size / 2))
        resting = bool(abs(center[2] - s) <= 0.005)
        released = True
        for contact in self._data.contact[: self._data.ncon]:
            if gid in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == gid else contact.geom1
                if other != self._floor_geom_id:
                    released = False
                    break
        still = bool(np.linalg.norm(self._data.joint(f'object_joint_{i}').qvel[:3]) <= 0.02)
        return {'inside': inside, 'resting': resting, 'released': released, 'still': still}

    def post_compilation_objects(self):
        self._floor_geom_id = self._model.geom('floor').id
        # Cube geom IDs.
        self._cube_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in cube_geoms] for cube_geoms in self._cube_geoms_list
        ]
        self._cube_target_mocap_ids = [
            self._model.body(f'object_target_{i}').mocapid[0] for i in range(self._num_cubes)
        ]
        self._cube_target_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in cube_target_geoms]
            for cube_target_geoms in self._cube_target_geoms_list
        ]

    def initialize_episode(self):
        self._success_hold_steps = 0
        # Set cube colors.
        for i in range(self._num_cubes):
            for gid in self._cube_geom_ids_list[i]:
                self._model.geom(gid).rgba = self._cube_colors[i]
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        self.initialize_arm()
        goal_xyzs = None if self._is_stack_anywhere_task() else self.cur_task_info['goal_xyzs'].copy()
        for _ in range(100):
            self._randomize_blocks()
            if self._is_stack_anywhere_task():
                success = all(self._compute_stack_anywhere_successes())
            else:
                block_xyzs = np.array([self._data.joint(f'object_joint_{i}').qpos[:3] for i in range(self._num_cubes)])
                success = np.all(np.linalg.norm(block_xyzs - goal_xyzs, axis=1) <= 0.04)
            if not success and not self._near_goal_marker():
                break
        if not self._is_stack_anywhere_task():
            for i in range(self._num_cubes):
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()

        # Forward kinematics to update site positions.
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def _compute_target_successes(self):
        """Compute target-mocap successes."""
        if self._goal_marker_size is not None:
            return [all(self.goal_marker_conditions(i).values()) for i in range(self._num_cubes)]
        cube_successes = []
        for i in range(self._num_cubes):
            obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = self._data.mocap_pos[self._cube_target_mocap_ids[i]]
            if np.linalg.norm(obj_pos - tar_pos) <= 0.04:
                cube_successes.append(True)
            else:
                cube_successes.append(False)

        return cube_successes

    def _compute_stack_anywhere_successes(self):
        block_xyzs = np.array([self._data.joint(f'object_joint_{i}').qpos[:3] for i in range(self._num_cubes)])
        order = np.argsort(block_xyzs[:, 2])
        sorted_xyzs = block_xyzs[order]
        target_zs = self._cube_size * (2 * np.arange(self._num_cubes) + 1)

        xy_success = np.all(np.linalg.norm(block_xyzs[:, :2] - sorted_xyzs[0, :2], axis=1) <= 0.04)
        z_success = np.all(np.abs(sorted_xyzs[:, 2] - target_zs) <= 0.03)
        success = bool(xy_success and z_success)
        return [success] * self._num_cubes

    def _compute_successes(self):
        """Compute object successes."""
        if self._is_stack_anywhere_task():
            return self._compute_stack_anywhere_successes()
        return self._compute_target_successes()

    def post_step(self):
        block_xyzs = np.array([self._data.joint(f'object_joint_{i}').qpos[:3] for i in range(self._num_cubes)])
        self._healthy = self._positions_in_workspace(block_xyzs)

        # Check if the cubes are in the target positions.
        cube_successes = self._compute_successes() if self._healthy else [False] * self._num_cubes
        self._success = self._healthy and all(cube_successes)
        if self._goal_marker_size is not None:
            # Strict success must hold for `success_hold_time` seconds of control steps.
            self._success_hold_steps = self._success_hold_steps + 1 if self._success else 0
            self._success = self._success_hold_steps * self._control_timestep >= self._success_hold_time - 1e-9

        # Adjust the colors of the cubes based on success.
        show_targets = not self.cur_task_info.get('hide_targets', False)
        for i in range(self._num_cubes):
            if self._visualize_info and show_targets:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

            if self._visualize_info and cube_successes[i]:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_success_colors[i, :3]
            else:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]

    def add_object_info(self, ob_info):
        # Cube positions and orientations.
        for i in range(self._num_cubes):
            ob_info[f'privileged/block_{i}_pos'] = self._data.joint(f'object_joint_{i}').qpos[:3].copy()
            ob_info[f'privileged/block_{i}_quat'] = self._data.joint(f'object_joint_{i}').qpos[3:].copy()
            ob_info[f'privileged/block_{i}_yaw'] = np.array(
                [lie.SO3(wxyz=self._data.joint(f'object_joint_{i}').qpos[3:]).compute_yaw_radians()]
            )
        if self._goal_marker_size is not None:
            # Inside, resting, released, still; and how long they have held.
            ob_info['privileged/goal_conditions'] = np.array(list(self.goal_marker_conditions(0).values()))
            ob_info['privileged/success_hold_steps'] = np.array([self._success_hold_steps])

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0

            ob_info = self.compute_ob_info()
            ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
            ]
            for i in range(self._num_cubes):
                ob.extend(
                    [
                        (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                        ob_info[f'privileged/block_{i}_quat'],
                        np.cos(ob_info[f'privileged/block_{i}_yaw']),
                        np.sin(ob_info[f'privileged/block_{i}_yaw']),
                    ]
                )

            return np.concatenate(ob)
