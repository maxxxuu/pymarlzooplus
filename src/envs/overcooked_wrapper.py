import numpy as np
from gym import ObservationWrapper
from gym.wrappers import TimeLimit as GymTimeLimit
import gym

from smac.env import MultiAgentEnv


class TimeLimitOvercooked(GymTimeLimit):

    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)

        assert max_episode_steps is not None, "'max_episode_steps' is None!"
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)

        self._elapsed_steps += 1
        info["TimeLimit.truncated"] = False  # There is no truncation in Overcooked
        if self._elapsed_steps >= self._max_episode_steps:
            done = True

        return observation, reward, done, info


class ObservationOvercooked(ObservationWrapper):
    """
    Observation wrapper that fixes the order of agents' observations.
    """

    def __init__(self, env):
        super(ObservationOvercooked, self).__init__(env)

        self.observation_space = env.observation_space.shape
        self._env = env
        self.other_agent_idx = None
        self.agent_policy_idx = None

    def observation(self, observation):
        if self._env._elapsed_steps == 0:  # Called from reset()
            # Get agents' ids to fix their observations and actions order
            self.other_agent_idx = observation['other_agent_env_idx']
            self.agent_policy_idx = 1 - self.other_agent_idx

        # Fix the order of observations, always 'policy_agent_idx' corresponds to agent 0
        assert self.agent_policy_idx == 1 - self.other_agent_idx
        assert self.other_agent_idx == observation['other_agent_env_idx']
        observation = [observation['both_agent_obs'][self.agent_policy_idx],
                       observation['both_agent_obs'][self.other_agent_idx]]

        return observation


OVERCOOKED_KEY_CHOICES = ["random3",
                          "random0",
                          "unident",
                          "soup_coordination",
                          "small_corridor",
                          "simple_tomato",
                          "simple_o_t",
                          "simple_o",
                          "schelling_s",
                          "schelling",
                          "m_shaped_s",
                          "long_cook_time",
                          "large_room",
                          "forced_coordination_tomato",
                          "forced_coordination",
                          "cramped_room_tomato",
                          "cramped_room_o_3orders",
                          "cramped_room",
                          "cramped_corridor",
                          "counter_circuit_o_1order",
                          "counter_circuit",
                          "corridor",
                          "coordination_ring",
                          "centre_objects",
                          "centre_pots",
                          "asymmetric_advantages",
                          "asymmetric_advantages_tomato",
                          "bottleneck"
                          ]
OVERCOOKED_REWARD_TYPE_CHOICES = ["shaped", "sparse"]


class _OvercookedWrapper(MultiAgentEnv):

    def __init__(self,
                 key,
                 time_limit,
                 seed,
                 reward_type):

        # Check key validity
        assert key in OVERCOOKED_KEY_CHOICES, \
            f"Invalid 'key': {key}! \nChoose one of the following: \n{OVERCOOKED_KEY_CHOICES}"
        # Check time_limit validity
        assert isinstance(time_limit, int), \
            f"Invalid time_limit type: {type(time_limit)}, 'time_limit': {time_limit}, is not 'int'!"
        # Check reward_type validity
        assert reward_type in OVERCOOKED_REWARD_TYPE_CHOICES, \
            f"Invalid 'reward_type': {reward_type}! \nChoose one of the following: \n{OVERCOOKED_REWARD_TYPE_CHOICES}"

        self.key = key
        self._seed = seed
        self.reward_type = reward_type

        # Gym make
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        mdp = OvercookedGridworld.from_layout_name(self.key)
        base_env = OvercookedEnv.from_mdp(mdp, horizon=time_limit)
        self.original_env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

        # Use the wrappers for handling the time limit and the environment observations properly.
        self.episode_limit = time_limit
        self.n_agents = 2  # Always 2 agents
        self._env = TimeLimitOvercooked(self.original_env, max_episode_steps=self.episode_limit)
        self._env = ObservationOvercooked(self._env)

        # Define the observation space
        self.observation_space = self._env.observation_space

        # Define the action space
        self.action_space = self._env.action_space.n

        # Placeholders
        self._obs = None
        self._info = None

        # By setting the "seed" in "np.random.seed" in "src/main.py" we control the randomness of the environment.
        self._seed = seed

        # Needed for rendering
        import cv2
        self.cv2 = cv2

    def step(self, actions):
        """ Returns reward, terminated, info """

        # Fix the order of actions, always 'policy_agent_idx' corresponds to agent 0
        actions = [int(a) for a in actions]
        if self._env.agent_policy_idx == 1:
            actions = actions[::-1]  # reverse the order

        # Make the environment step
        self._obs, reward, done, self._info = self._env.step(actions)

        if self.reward_type == "shaped":
            assert type(self._info['shaped_r_by_agent']) is list, \
                "'self._info['shaped_r_by_agent']' is not a list! " + \
                f"'self._info['shaped_r_by_agent']': {self._info['shaped_r_by_agent']}"
            reward = sum(self._info['shaped_r_by_agent'])
        # else: the other option is the sum of sparse rewards which the default 'reward'

        # Keep only 'TimeLimit.truncated' in 'self._info'
        self._info = {"TimeLimit.truncated": self._info["TimeLimit.truncated"]}

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

    def reset(self):
        """ Returns initial observations and states """

        self._obs = self._env.reset()

        return self.get_obs(), self.get_state()

    def render(self):
        image = self._env.render()
        image = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
        self.cv2.imshow("Overcooked", image)
        self.cv2.waitKey(1)

    def close(self):
        self._env.close()

    def seed(self):
        return self._seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
