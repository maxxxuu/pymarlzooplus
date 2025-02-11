import random

import numpy as np
from gym import ObservationWrapper
from gym.wrappers import TimeLimit as GymTimeLimit
import gym

from smac.env import MultiAgentEnv


# Wraps the original environment and adds the extra var "elapsed_time" to keep track of when an episode starts

class TimeLimitPressurePlate(GymTimeLimit):

    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)

        assert max_episode_steps is not None, "'max_episode_steps' is None!"
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"
        observations, rewards, terminations, infos = self.env.step(action)

        self._elapsed_steps += 1
        infos["TimeLimit.truncated"] = False  # dummy var, there is no truncation in PressurePlate
        if self._elapsed_steps >= self._max_episode_steps:
            terminations = [True for _ in terminations]

        return observations, rewards, terminations, infos


class ObservationPressurePlate(ObservationWrapper):
    """
    Observation wrapper that fixes the order of agents' observations.
    """

    def __init__(self, env):
        super(ObservationPressurePlate, self).__init__(env)
        self._env = env
        self.observation_space = self._env.observation_space[0].shape

    def observation(self, observation):
        return [
            obs for obs in observation
        ]


PRESSUREPLATE_KEY_CHOICES = [
    "pressureplate-linear-4p-v0",
    "pressureplate-linear-5p-v0",
    "pressureplate-linear-6p-v0"
]

PRESSUREPLATE_N_AGENTS_CHOICES = [4, 5, 6]


class _PressurePlateWrapper(MultiAgentEnv):

    def __init__(
            self,
            key,
            time_limit=500,
            seed=1,
    ):

        # Check key validity
        assert key in PRESSUREPLATE_KEY_CHOICES, \
            f"Invalid 'key': {key}! \nChoose one of the following: \n{PRESSUREPLATE_KEY_CHOICES}"
        # Check time_limit validity
        assert isinstance(time_limit, int), \
            f"Invalid time_limit type: {type(time_limit)}, 'time_limit': {time_limit}, is not 'int'!"

        self.key = key
        self._seed = seed

        # Placeholders
        self.original_env = None
        self.episode_limit = None
        self._env = None
        self._obs = None
        self._info = None
        self.observation_space = None
        self.action_space = None

        ## Gym make
        # base env sourced by gym.make with all its args
        from pressureplate.environment import PressurePlate
        self.original_env = gym.make(f"{key}")
        self._seed = self.original_env.seed(self._seed)

        # Use the wrappers for handling the time limit and the environment observations properly.
        self.n_agents = self.original_env.n_agents
        self.episode_limit = time_limit
        # now create the wrapped env
        self._env = TimeLimitPressurePlate(self.original_env, max_episode_steps=self.episode_limit)
        self._env = ObservationPressurePlate(self._env)

        # Define the observation space
        self.observation_space = self._env.observation_space
        # Define the action space
        self.action_space = self._env.action_space[0].n
        # Placeholders
        self._obs = None
        self._info = None

        # Needed for rendering
        import cv2
        self.cv2 = cv2

    def step(self, actions):
        """ Returns reward, terminated, info """

        # From torch.tensor to int
        actions = [int(a) for a in actions]

        # Make the environment step
        self._obs, rewards, terminations, self._info = self._env.step(actions)

        # Add all rewards together. 'rewards' is a list
        reward = sum(rewards)

        # Keep only 'TimeLimit.truncated' in 'self._info'
        self._info = {"TimeLimit.truncated": self._info["TimeLimit.truncated"]}

        # The episode ends when all agents have reached their positions ("terminations" are all True) or
        # "self._elapsed_steps >= self._max_episode_steps" is True
        done = all(terminations)

        return float(reward), done, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.observation_space[0]

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state """

        assert len(self.observation_space) == 1, \
            f"'self.observation_space' has not only one dimension! \n'self.observation_space': {self.observation_space}"

        return self.n_agents * self.observation_space[0]

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)

        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id (both agents have the same action space) """
        return self.action_space * [1]  # 1 indicates availability of action

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.action_space

    def sample_actions(self):
        return random.choices(range(0, self.get_total_actions()), k=self.n_agents)

    def reset(self, seed=None):
        """ Returns initial observations and states """

        # Control seed
        if seed is not None:
            self._seed = seed
            self._seed = self.original_env.seed(self._seed)

        self._obs = self._env.reset()
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
