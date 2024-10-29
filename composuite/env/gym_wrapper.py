import numpy as np
from gym import spaces

from robosuite.wrappers.gym_wrapper import GymWrapper

class GymWrapper(GymWrapper):
    """Extends the robosuite Gym wrapper which mimics many of the 
    required functionalities of the Wrapper class found in the gym.core module.
    """

    def __init__(self, env, use_camera_obs=False, keys=None, info_keys=None):
        """Args:
            env (MujocoEnv): The environment to wrap.
            keys (None, list of str): If provided, each observation will
                consist of concatenated keys from the wrapped environment's
                observation dictionary. If None, all the keys from the
                compositional environment will be added.
            info_keys (None, list of str): If provided, each observation dict key
                will be added to the info dict of the gym env
        """
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        self.use_camera_obs = use_camera_obs

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add obstacle obs if requested
            if hasattr(self.env, "use_obstacle_obs") and self.env.use_obstacle_obs:
                keys += ["obstacle-state"]
            # Add goal obs if requested
            if hasattr(self.env, "use_goal_obs") and self.env.use_goal_obs:
                keys += ["goal-state"]
            # Add task obs if requested
            if self.env.use_task_id_obs:
                # keys += ["task-state"]
                keys += ["object_id", "robot_id", "obstacle_id", "subtask_id"]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys
        self.info_keys = info_keys

        # Gym specific attributes
        self.env.spec = None
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        self.observation_positions = {}
        idx = 0
        for key in self.keys:
            self.observation_positions[key] = np.arange(idx, obs[key].shape[0] + idx)
            idx += obs[key].shape[0]

        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)


    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)
        if self.use_camera_obs:
            for cam_name in self.env.camera_names:
                info[cam_name] = ob_dict[f"{cam_name}_image"]
                del ob_dict[f"{cam_name}_image"]
        if self.info_keys:
            for key in self.info_keys:
                info[key] = ob_dict[key]
        return self._flatten_obs(ob_dict), reward, terminated, False, info

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            2-tuple:
                - (np.array) flattened observations from the environment
                - (dict) an empty dictionary, as part of the standard return format
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        info = {}
        if self.use_camera_obs:
            for cam_name in self.env.camera_names:
                info[cam_name] = ob_dict[f"{cam_name}_image"]
                del ob_dict[f"{cam_name}_image"]
        return self._flatten_obs(ob_dict), info