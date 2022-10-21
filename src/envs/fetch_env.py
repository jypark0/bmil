import numpy as np
from gym.envs.robotics import FetchPickAndPlaceEnv, FetchPushEnv


class RandomizeMixIn:
    def _create_discrete_grid(self):
        # Initialize discrete grid cells 5x5x5
        self.cells_per_dim = 5

        x = np.linspace(
            self.gripper_min[0], self.gripper_max[0], self.cells_per_dim + 1
        )
        # Get midpoint
        x = (x[1:] + x[:-1]) / 2

        y = np.linspace(
            self.gripper_min[1], self.gripper_max[1], self.cells_per_dim + 1
        )
        # Get midpoint
        y = (y[1:] + y[:-1]) / 2

        z = np.linspace(
            self.gripper_min[2], self.gripper_max[2], self.cells_per_dim + 1
        )
        # Get midpoint
        z = (z[1:] + z[:-1]) / 2

        discrete_grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

        return discrete_grid

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize gripper
        if self.random_gripper:
            # Table boundaries (1.05-1.55, 0.4-1.1, 0.4)
            # Initial_gripper_xpos (1.342, 0.749, 0.535)
            new_gripper_pos = self.initial_gripper_xpos + self.np_random.uniform(
                low=self.gripper_min, high=self.gripper_max
            )
            # Minor adjustment
            new_gripper_pos[2] += 0.02
            self.sim.data.set_mocap_pos("robot0:mocap", new_gripper_pos)
            for _ in range(10):
                self.sim.step()

            # Make sure gripper is closed if needed
            if self.block_gripper:
                self.sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", 0.0)
                self.sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", 0.0)
                self.sim.forward()

        elif self.init_cell is not None:
            raise NotImplementedError("doesn't work  for now")
            # new_gripper_pos = (
            #     self.initial_gripper_xpos + self.discrete_grid[self.init_cell]
            # )
            # cell_size = (self.gripper_max - self.gripper_min) / self.cells_per_dim
            # new_gripper_pos += self.np_random.uniform(
            #     low=-cell_size / 2, high=cell_size / 2
            # )
            # # Minor adjustment
            # self.sim.data.set_mocap_pos("robot0:mocap", new_gripper_pos)
            # for _ in range(10):
            #     self.sim.step()
            #
            # # Make sure gripper is closed if needed
            # if self.block_gripper:
            #     self.sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", 0.0)
            #     self.sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", 0.0)
            #     self.sim.forward()

        def _fixed_object():
            object_xpos = self.initial_gripper_xpos[:2] + np.array(
                [self.obj_range / 4, self.obj_range / 4]
            )
            # object_xpos = self.initial_gripper_xpos[:2] + np.array(
            #     [self.obj_range / 4, -self.obj_range / 4]
            # )
            # object_xpos = self.initial_gripper_xpos[:2] + np.array(
            #     [self.obj_range / 4, 0]
            # )

            return object_xpos

        def _random_object():
            object_xpos = self.initial_gripper_xpos[:2]
            # Get current gripper pos
            if self.random_gripper:
                while (
                    np.linalg.norm(
                        object_xpos - self.sim.data.get_site_xpos("robot0:grip")[:2]
                    )
                    < 0.1
                ):
                    object_xpos = self.initial_gripper_xpos[
                        :2
                    ] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            else:
                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                    object_xpos = self.initial_gripper_xpos[
                        :2
                    ] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            return object_xpos

        if self.has_object:
            if self.random_object:
                object_xpos = _random_object()
            else:
                object_xpos = _fixed_object()
            self.initial_object_pos = object_xpos.copy()

            # Modify object xpos
            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos("object0:joint", object_qpos)

        self.sim.forward()

        return True

    def _sample_goal(self):
        def _fixed_goal():
            goal = self.initial_gripper_xpos[:3] + np.array(
                [self.target_range / 2, self.target_range / 2, self.target_range / 2]
            )
            if self.has_object:
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air:
                    goal[2] += 0.2

            return goal

        def _random_goal():
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            if self.has_object:
                goal += self.target_offset
                goal[2] = self.height_offset
                if self.target_in_the_air and self.np_random.uniform() < 0.5:
                    goal[2] += self.np_random.uniform(0, 0.45)

            return goal

        # If called from __init__ before seed has been set
        if not hasattr(self, "random_goal"):
            return _fixed_goal().copy()

        # Sample random goal
        if self.random_goal:
            goal = _random_goal()
        # Use fixed goal position
        else:
            goal = _fixed_goal()
        self.initial_goal_pos = goal.copy()

        return goal.copy()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        if self.terminate_on_success:
            done = self._is_success(obs["achieved_goal"], self.goal)
        else:
            done = False
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        return obs, reward, done, info


class MyFetchPickAndPlaceEnv(RandomizeMixIn, FetchPickAndPlaceEnv):
    def __init__(
        self,
        reward_type="sparse",
        random_gripper=False,
        random_object=True,
        random_goal=True,
        terminate_on_success=False,
        init_cell=None,
    ):
        assert (
            int(random_gripper) + int(init_cell is not None) <= 1
        ), "Only one of random_gripper, init_cell should be given"

        FetchPickAndPlaceEnv.__init__(self, reward_type=reward_type)

        self.random_gripper = random_gripper
        self.random_object = random_object
        self.random_goal = random_goal
        self.terminate_on_success = terminate_on_success

        # self.gripper_min = np.array([-0.15, -0.15, -0.1])
        # self.gripper_max = np.array([0.15, 0.15, 0.165])
        # Don't vary z
        self.gripper_min = np.array([-0.15, -0.15, 0])
        self.gripper_max = np.array([0.15, 0.15, 0])

        # Check that init_cell is valid
        self.discrete_grid = self._create_discrete_grid()
        if init_cell is not None and init_cell >= len(self.discrete_grid):
            raise ValueError(
                f"init_cell index {init_cell} is invalid (grid size: {len(self.discrete_grid)})"
            )
        self.init_cell = init_cell


class MyFetchPushEnv(RandomizeMixIn, FetchPushEnv):
    def __init__(
        self,
        reward_type="sparse",
        random_gripper=False,
        random_object=True,
        random_goal=True,
        terminate_on_success=False,
        init_cell=None,
    ):
        assert (
            int(random_gripper) + int(init_cell is not None) <= 1
        ), "Only one of random_gripper, init_cell should be given"
        FetchPushEnv.__init__(self, reward_type=reward_type)

        self.random_gripper = random_gripper
        self.random_object = random_object
        self.random_goal = random_goal
        self.terminate_on_success = terminate_on_success

        # self.gripper_min = np.array([-0.15, -0.15, 0])
        # self.gripper_max = np.array([0.15, 0.15, 0.28])
        # Don't vary z
        self.gripper_min = np.array([-0.15, -0.15, 0])
        self.gripper_max = np.array([0.15, 0.15, 0])

        # Check that init_cell is valid
        self.discrete_grid = self._create_discrete_grid()
        if init_cell is not None and init_cell >= len(self.discrete_grid):
            raise ValueError(
                f"init_cell index {init_cell} is invalid (grid size: {len(self.discrete_grid)})"
            )
        self.init_cell = init_cell
