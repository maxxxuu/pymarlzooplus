import gym
from gym import ObservationWrapper
from gym.wrappers import TimeLimit as GymTimeLimit
from gym.envs.registration import register

from smac.env import MultiAgentEnv


class TimeLimitCT(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)

        assert max_episode_steps is not None, "'max_episode_steps' is None!"
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, actions):
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"

        observations, rewards, done, info = self.env.step(actions)
        done = bool(done)  # From int to bool

        self._elapsed_steps += 1
        info = {"TimeLimit.truncated": False}

        if self._elapsed_steps >= self._max_episode_steps:
            done = True

        return observations, rewards, done, info


class ObservationCT(ObservationWrapper):

    def __init__(self, env):
        super(ObservationCT, self).__init__(env)

        self._env = env

    def step(self, actions):
        observations, rewards, done, info = self.env.step(actions)
        return self.observation(observations), rewards, done, info

    def observation(self, observations):
        return observations


class _CaptureTargetWrapper(MultiAgentEnv):

    def __init__(self, key, seed, **kwargs):

        # Check time_limit validity
        assert 'time_limit' in list(kwargs.keys()), 'time_limit not provided!'
        time_limit = kwargs['time_limit']
        assert isinstance(time_limit, int), \
            f"Invalid time_limit type: {type(time_limit)}, 'time_limit': {time_limit}, is not 'int'!"

        # Fix kwargs time_limit, since the environment defines this arg as 'terminate_step'
        kwargs['terminate_step'] = time_limit
        del kwargs['time_limit']

        # Import and register
        from capture_target_ai_py.environment import CaptureTarget
        self.gym_registration()

        self.original_env = gym.make(f"{key}", **kwargs)
        self.episode_limit = self.original_env.terminate_step

        self._env = TimeLimitCT(self.original_env, max_episode_steps=self.episode_limit)
        self._env = ObservationCT(self._env)

        self.n_agents = self._env.n_agent
        self._obs = None
        self._info = {"TimeLimit.truncated": False}
        self._seed = seed
        self._obs_size = self._env.obs_size[0]
        self.action_space = self._env.n_action[0]

        # Check if n_agents is 2
        assert self.n_agents == 2
        # Check if obs_size is the same for every agent
        assert all([self._obs_size == obs_size for obs_size in self._env.obs_size])
        # Check if n_action is the same for every agent
        assert all([self.action_space == n_action for n_action in self._env.n_action])

    @staticmethod
    def gym_registration():
        register(
            id="CaptureTarget-6x6-1t-2a-v0",
            entry_point="capture_target_ai_py.environment:CaptureTarget",
            kwargs={
                "n_target": 1,
                "n_agent": 2,
                "grid_dim": [6, 6]
            },
        )

    def step(self, actions):
        """ Returns reward, terminated, info """

        # From torch.tensor to int
        actions = [int(a) for a in actions]

        # Make the environment step
        self._obs, rewards, done, info = self._env.step(actions)

        # Add all rewards together. 'rewards' is a list
        rewards = sum(rewards)

        return float(rewards), done, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self._obs_size

    def get_state(self):
        # The state is a vector with the observations of all agents flattened
        return self._obs.flatten()

    def get_state_size(self):
        """ Returns the flatten shape of the state"""
        return self.n_agents * self._obs_size

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return self.action_space * [1]  # 1 indicates availability of actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.action_space

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
