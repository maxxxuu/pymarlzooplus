import numpy as np
from gym import ObservationWrapper
from gym.wrappers import TimeLimit as GymTimeLimit

from smac.env import MultiAgentEnv
from utils.image_encoder import ImageEncoder


class TimeLimitPZ(GymTimeLimit):
    def __init__(self, env, key, max_episode_steps):
        super().__init__(env)

        self._max_episode_steps = max_episode_steps
        self._key = key
        self._elapsed_steps = None  # This is initialized to 0 by GymTimeLimit when reset() is calling.
        self._obs_wrapper = None

    def step(self, actions):
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"

        if self._key == "entombed_cooperative_v3":

            # In this step, agent 1 is moving
            previous_observations_, previous_rewards_, previous_terminations_, previous_truncations_, _ = \
                self.env.step(actions)
            previous_terminations__all = any([termination for termination in previous_terminations_.values()])
            previous_truncations__all = any([truncation for truncation in previous_truncations_.values()])

            if previous_terminations__all is False and previous_truncations__all is False:

                # In this step, agent 2 is moving
                observations_, rewards_, terminations_, truncations_, _ = self.env.step(actions)
                terminations__all = any([termination for termination in terminations_.values()])
                truncations__all = any([truncation for truncation in truncations_.values()])

                if terminations__all is False and truncations__all is False:

                    ## Perform no action 2 times in order to sync obs and actions
                    no_actions = {'first_0': 0, 'second_0': 0}

                    # First no action
                    previous_observations, previous_rewards, previous_terminations, previous_truncations, _ = \
                        self.env.step(no_actions)
                    previous_obs = list(previous_observations.values())[0]
                    previous_terminations_all = any([termination for termination in previous_terminations.values()])
                    previous_truncations_all = any([truncation for truncation in previous_truncations.values()])

                    if previous_terminations_all is False and previous_truncations_all is False:

                        # Second no action
                        observations, rewards, terminations, truncations, infos = self.env.step(no_actions)
                        current_obs = list(observations.values())[1]

                        # Get the combined obs
                        observations = \
                            self._obs_wrapper.entombed_cooperative_v3_get_combined_images(previous_obs, current_obs)

                        rewards = {reward_key: sum([reward, reward_, previous_reward, previous_reward_])
                                   for (reward_key, reward, reward_, previous_reward, previous_reward_) in
                                   zip(rewards.keys(),
                                       rewards.values(),
                                       rewards_.values(),
                                       previous_rewards.values(),
                                       previous_rewards_.values())
                                   }

                        self._elapsed_steps += 4
                    else:  # Third step termination case
                        # In this case, we don't really care about observations
                        # since it's the last step (due to termination)
                        observations = previous_observations
                        rewards = {reward_key: sum([reward_, previous_reward, previous_reward_])
                                   for (reward_key, reward_, previous_reward, previous_reward_) in
                                   zip(rewards_.keys(),
                                       rewards_.values(),
                                       previous_rewards.values(),
                                       previous_rewards_.values())
                                   }
                        terminations = previous_terminations
                        truncations = previous_truncations
                        self._elapsed_steps += 3
                else:  # Second step termination case
                    # In this case, we don't really care about observations
                    # since it's the last step (due to termination)
                    observations = observations_
                    rewards = {reward_key: sum([reward_, previous_reward_])
                               for (reward_key, reward_, previous_reward_) in
                               zip(rewards_.keys(),
                                   rewards_.values(),
                                   previous_rewards_.values())
                               }
                    terminations = terminations_
                    truncations = truncations_
                    self._elapsed_steps += 2
            else:  # First step termination case
                # In this case, we don't really care about observations
                # since it's the last step (due to termination)
                observations = previous_observations_
                rewards = previous_rewards_
                terminations = previous_terminations_
                truncations = previous_truncations_
                self._elapsed_steps += 1

        elif self._key == "space_invaders_v2":
            # After extensive investigation, we found out that the "move" actions should be applied at the first step
            # and the "fire" actions at the second step, in order to apply the actions consistently.
            move_actions = {'first_0': actions['first_0'] if actions['first_0'] != 1 else 0,
                            'second_0': actions['second_0'] if actions['second_0'] != 1 else 0
                            }
            fire_actions = {'first_0': actions['first_0'] if actions['first_0'] == 1 else 0,
                            'second_0': actions['second_0'] if actions['second_0'] == 1 else 0
                            }
            # Perform the decided actions in order to get the state which is not full due to flickering
            previous_observations, previous_rewards, previous_terminations, previous_truncations, _ = \
                self.env.step(move_actions)
            previous_obs = list(previous_observations.values())[0]
            previous_terminations_all = any([termination for termination in previous_terminations.values()])
            previous_truncations_all = any([truncation for truncation in previous_truncations.values()])

            if previous_terminations_all is False and previous_truncations_all is False:
                # Perform no action and get the next state which is not full due to flickering
                observations, rewards, terminations, truncations, infos = self.env.step(fire_actions)
                current_obs = list(observations.values())[1]

                # Combine the two states to get the full state after applying the actions
                observations = self._obs_wrapper.space_invaders_v2_get_combined_images(previous_obs, current_obs)

                rewards = {reward_key: sum([reward, previous_reward])
                           for (reward_key, reward, previous_reward) in
                           zip(rewards.keys(),
                               rewards.values(),
                               previous_rewards.values())
                           }

                self._elapsed_steps += 2

            else:  # First step termination case
                # In this case, we don't really care about observations
                # since it's the last step (due to termination)
                observations = previous_observations
                rewards = previous_rewards
                terminations = previous_terminations
                truncations = previous_truncations
                self._elapsed_steps += 1

        else:
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            self._elapsed_steps += 1

        info = {"TimeLimit.truncated": any([truncation for truncation in truncations.values()])}

        if self._elapsed_steps >= self._max_episode_steps:
            terminations = {key: True for key in terminations.keys()}

        return observations, rewards, terminations, truncations, info


class ObservationPZ(ObservationWrapper):
    """
    Observation wrapper for converting images to vectors (using a pretrained image encoder) or
    for preparing images to be fed to a CNN.
    """

    def __init__(self,
                 env,
                 partial_observation,
                 trainable_cnn,
                 image_encoder,
                 image_encoder_batch_size,
                 image_encoder_use_cuda,
                 centralized_image_encoding,
                 given_observation_space):

        # Initialize 'ObservationWrapper'
        super().__init__(env)

        # Initialize 'ImageEncoder' and get useful attributes
        self.image_encoder = ImageEncoder(
                              "env",
                              centralized_image_encoding,
                              trainable_cnn,
                              image_encoder,
                              image_encoder_batch_size,
                              image_encoder_use_cuda
                              )
        self.print_info = self.image_encoder.print_info
        self.observation_space = self.image_encoder.observation_space

        self.partial_observation = partial_observation
        self.original_observation_space = self.env.observation_space(self.env.possible_agents[0])
        self.original_observation_space_shape = self.original_observation_space.shape
        self.is_image = len(self.original_observation_space_shape) == 3 and self.original_observation_space_shape[2] == 3
        assert self.is_image, f"Only images are supported, shape: {self.original_observation_space_shape}"

        self.given_observation_space = given_observation_space
        if given_observation_space is not None:
            self.observation_space = given_observation_space
        assert not (given_observation_space is None and centralized_image_encoding is True)
        assert not (given_observation_space is not None and centralized_image_encoding is False)

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return self.observation(observations), rewards, terminations, truncations, infos

    def observation(self, observations):
        return self.image_encoder.observation(observations)

    @staticmethod
    def replace_color(image_, target_color, replacement_color):
        # Find all pixels matching the target color
        matches = np.all(image_ == target_color, axis=-1)
        # Replace these pixels with the replacement color
        image_[matches] = replacement_color

        return image_

    def space_invaders_v2_get_combined_images(self, image_a, image_b, sensitivity=0):
        # We should remove the red ship, and the two agents from image A in order to get
        # just their final position from image B, otherwise artifacts will be created due
        # to the minor movements of these objects.
        red_ship_rgb_values = [181, 83, 40]
        agent_1_rgb_values = [50, 132, 50]
        agent_2_rgb_values = [162, 134, 56]
        image_a = self.replace_color(image_a, red_ship_rgb_values, [0, 0, 0])
        image_a = self.replace_color(image_a, agent_1_rgb_values, [0, 0, 0])
        image_a = self.replace_color(image_a, agent_2_rgb_values, [0, 0, 0])
        # Calculate the absolute difference between images
        diff = self.cv2.absdiff(image_a, image_b)
        # Convert the difference to grayscale in order to handle single threshold for all channels
        diff_gray = self.cv2.cvtColor(diff, self.cv2.COLOR_RGB2GRAY)
        # Mask for common areas: where the difference is less than or equal to sensitivity
        common_mask = np.where(diff_gray <= sensitivity, 255, 0).astype(np.uint8)
        # Mask for differences: where the difference is greater than sensitivity
        difference_mask = np.where(diff_gray > sensitivity, 255, 0).astype(np.uint8)
        # Create a 3-channel mask for common and difference areas
        common_mask_3channel = self.cv2.cvtColor(common_mask, self.cv2.COLOR_GRAY2RGB)
        difference_mask_3channel = self.cv2.cvtColor(difference_mask, self.cv2.COLOR_GRAY2RGB)
        # Extract common areas using common mask
        common_areas = self.cv2.bitwise_and(image_a, common_mask_3channel)
        # Extract differences from both images
        differences_from_a = self.cv2.bitwise_and(image_a, difference_mask_3channel)
        differences_from_b = self.cv2.bitwise_and(image_b, difference_mask_3channel)
        # Combine common areas with differences from both images
        combined_image = self.cv2.add(common_areas, differences_from_a)
        combined_image = self.cv2.add(combined_image, differences_from_b)
        # Create partial obs of agent 1 by removing agent 2 from the combined image
        agent_1_obs_ = self.replace_color(combined_image.copy(), agent_2_rgb_values, [0, 0, 0])
        # Create partial obe of agent 2 by removing agent 1 from the combined image
        agent_2_obs_ = self.replace_color(combined_image.copy(), agent_1_rgb_values, [0, 0, 0])

        if self.partial_observation is True:
            obs = {'first_0': agent_1_obs_, 'second_0': agent_2_obs_}
        else:
            obs = {'first_0': combined_image.copy(), 'second_0': combined_image.copy()}

        return obs

    def entombed_cooperative_v3_get_combined_images(self, image_a, image_b):
        # Define the RGB values of agent 1
        agent_1_rgb_values = np.array([232, 232, 74])
        # Define the RGB values of agent 2
        agent_2_rgb_values = np.array([197, 124, 238])
        # Find where image A has the specific RGB values of agent 1
        mask = np.all(image_a == agent_1_rgb_values, axis=-1)
        # Find where image B has the specific RGB value of agent 1
        mask_ = np.all(image_b == agent_1_rgb_values, axis=-1)
        # Find which is the image which illustrates agent 1
        if mask_.sum() > mask.sum():
            image_b = image_a
            mask = mask_
        # Replace the corresponding values in image B where the mask is True (that is where agent 1 is located)
        combined_image = image_b.copy()
        combined_image[mask] = agent_1_rgb_values
        # Create partial obs of agent 1 by removing agent 2 from the combined image
        black_rgb_values = [0, 0, 0]
        agent_1_obs_ = self.replace_color(combined_image.copy(), agent_1_rgb_values, black_rgb_values)
        # Create partial obe of agent 2 by removing agent 1 from the combined image
        agent_2_obs_ = self.replace_color(combined_image.copy(), agent_2_rgb_values, black_rgb_values)

        if self.partial_observation is True:
            obs = {'first_0': agent_1_obs_, 'second_0': agent_2_obs_}
        else:
            obs = {'first_0': combined_image.copy(), 'second_0': combined_image.copy()}

        return obs


class _PettingZooWrapper(MultiAgentEnv):
    def __init__(self,
                 key,
                 max_cycles,
                 seed,
                 render_mode="rgb_array",
                 partial_observation=False,
                 trainable_cnn=False,
                 image_encoder="ResNet18",
                 image_encoder_batch_size=1,
                 image_encoder_use_cuda=False,
                 centralized_image_encoding=False,
                 kwargs="",
                 given_observation_space=None):

        assert (partial_observation is False) or (key in ["entombed_cooperative_v3", "space_invaders_v2"]), \
            ("'partial_observation' should False when the selected game is other than "
             "'entombed_cooperative_v3' or 'space_invaders_v2'!")

        self.key = key
        self.max_cycles = max_cycles
        self._seed = seed
        self.render_mode = render_mode
        self.partial_observation = partial_observation
        self.trainable_cnn = trainable_cnn
        self._image_encoder = image_encoder
        self.image_encoder_batch_size = image_encoder_batch_size
        self.image_encoder_use_cuda = image_encoder_use_cuda
        self.centralized_image_encoding = centralized_image_encoding
        self._kwargs = kwargs
        self.given_observation_space = given_observation_space

        # Placeholders
        self.kwargs = None
        self.original_env = None
        self.episode_limit = None
        self.n_agents = None
        self._env = None
        self.__env = None
        self._obs = None
        self._info = None
        self.observation_space = None
        self.action_space = None
        self.action_prefix = None
        self.cv2 = None
        self.sum_rewards = None

        # Environment
        self.set_environment(self.key,
                             self.max_cycles,
                             self.render_mode,
                             self.partial_observation,
                             self.trainable_cnn,
                             self._image_encoder,
                             self.image_encoder_batch_size,
                             self.image_encoder_use_cuda,
                             self.centralized_image_encoding,
                             self._kwargs,
                             self.given_observation_space)

    def set_environment(self,
                        key,
                        max_cycles,
                        render_mode,
                        partial_observation,
                        trainable_cnn,
                        image_encoder,
                        image_encoder_batch_size,
                        image_encoder_use_cuda,
                        centralized_image_encoding,
                        kwargs,
                        given_observation_space):

        # Convert list of kwargs to dictionary
        self.kwargs = kwargs
        self.get_kwargs(max_cycles, render_mode)

        # Assign value to 'self.sum_rewards' based on the env key
        self.sum_rewards = True
        if key not in ["pistonball_v6", "cooperative_pong_v5", "entombed_cooperative_v3", "space_invaders_v2"]:
            # Only these environments refer to fully cooperation, thus we can sum the rewards.
            self.sum_rewards = False

        # Define the environment
        self.original_env = self.pettingzoo_make(key, self.kwargs)

        try:
            self.episode_limit = self.original_env.unwrapped.max_cycles
        except:
            self.episode_limit = self.original_env.unwrapped.env.max_cycles
        assert self.episode_limit > 1, "self.episode_limit should be > 1: {self.episode_limit}"

        self.n_agents = self.original_env.max_num_agents
        assert (self.n_agents == 2) or (key not in ["entombed_cooperative_v3", "space_invaders_v2"]), \
            ("Only 2 agents have been considered for 'entombed_cooperative_v3' and 'space_invaders_v2'! "
             "'n_agents': {}".format(self.n_agents))
        setattr(self.original_env, 'spec', None)  # Just for support with "ObservationWrapper"

        self.__env = TimeLimitPZ(self.original_env,
                                 key,
                                 max_episode_steps=self.episode_limit)
        # In case of "entombed_cooperative_v3" and "space_invaders_v2" games,
        # fix the "self.episode_limit" after passing it to the "TimeLimitPZ"
        if key == "entombed_cooperative_v3":
            # At each timestep, we apply 4 pettingzoo timesteps,
            # in order to synchronize actions and obs
            self.episode_limit = int(self.episode_limit / 4)
        elif key == "space_invaders_v2":
            # At each timestep, we apply 2 pettingzoo timesteps,
            # in order to synchronize actions and obs
            self.episode_limit = int(self.episode_limit / 2)

        self._env = ObservationPZ(self.__env,
                                  partial_observation,
                                  trainable_cnn,
                                  image_encoder,
                                  image_encoder_batch_size,
                                  image_encoder_use_cuda,
                                  centralized_image_encoding,
                                  given_observation_space)
        self.__env._obs_wrapper = self._env

        self._obs = None
        self._info = None

        # Define the observation space
        self.observation_space = self._env.observation_space

        ## Define the action space
        tmp_action_spaces_list = [self._env.action_space(possible_agent) for possible_agent in
                                  self._env.possible_agents]
        tmp_action_space = tmp_action_spaces_list[0]
        # Check that all agents have the same action space
        assert not isinstance(tmp_action_space, tuple), f"tmp_action_space: {tmp_action_space}"
        for agent_idx, tmp_action_space_ in enumerate(tmp_action_spaces_list[1:], start=1):
            assert tmp_action_space_ == tmp_action_space, \
                ("Difference in action spaces found between agents:\n"
                 f"agent=0 - action_space={tmp_action_space}, agent={agent_idx} - action_space={tmp_action_space_}")
        self.action_space = tmp_action_space.n
        self.action_prefix = [action_prefix for action_prefix in self._env.possible_agents]

        # Use cv2 for rendering
        import cv2
        self.cv2 = cv2

    def get_kwargs(self, max_cycles, render_mode):
        if isinstance(self.kwargs, list):
            if isinstance(self.kwargs[0], list):
                # Convert list to dict
                self.kwargs = {arg[0]: arg[1] for arg in self.kwargs}
            else:
                # Convert single arguments to dict
                assert isinstance(self.kwargs[0], str)
                tmp_kwargs = self.kwargs
                self.kwargs = {tmp_kwargs[0]: tmp_kwargs[1]}
        else:
            assert isinstance(self.kwargs, str), f"Unsupported kwargs type: {self.kwargs}"
            self.kwargs = {}

        if max_cycles is not None:
            self.kwargs["max_cycles"] = max_cycles
        if render_mode is not None:
            self.kwargs["render_mode"] = render_mode

    @staticmethod
    def pettingzoo_make(env_name, kwargs):
        if env_name == "pistonball_v6":
            try:
                from pettingzoo.butterfly import pistonball_v6
            except:
                raise ImportError("pettingzoo[butterfly] is not installed! "
                                  "\nInstall it running: \npip install 'pettingzoo[butterfly]'")

            if 'continuous' not in kwargs.keys():
                kwargs['continuous'] = False
            assert kwargs['continuous'] is False, "'continuous' argument should be True!"

            if 'n_pistons' in kwargs.keys():
                assert kwargs['n_pistons'] >= 4, \
                    "'n_pistons' argument must be greater than or equal to 4!"
                # Otherwise, the game stops almost immediately.

            return pistonball_v6.parallel_env(**kwargs)  # Parallel mode

        elif env_name == "cooperative_pong_v5":
            try:
                from pettingzoo.butterfly import cooperative_pong_v5
            except:
                raise ImportError("pettingzoo[butterfly] is not installed! "
                                  "\nInstall it running: \npip install 'pettingzoo[butterfly]'")

            return cooperative_pong_v5.parallel_env(**kwargs)

        elif env_name == "entombed_cooperative_v3":
            try:
                from pettingzoo.atari import entombed_cooperative_v3
            except:
                raise ImportError("pettingzoo[atari] is not installed! "
                                  "\nInstall it running: \npip install 'pettingzoo[atari]'")

            return entombed_cooperative_v3.parallel_env(**kwargs)

        elif env_name == "space_invaders_v2":
            try:
                from pettingzoo.atari import space_invaders_v2
            except:
                raise ImportError("pettingzoo[atari] is not installed! "
                                  "\nInstall it running: \npip install 'pettingzoo[atari]'")

            if 'alternating_control' not in kwargs.keys():
                kwargs['alternating_control'] = False
            assert kwargs['alternating_control'] is False, "'alternating_control' should be False!"

            return space_invaders_v2.parallel_env(**kwargs)

        elif env_name == "basketball_pong_v3":
            try:
                from pettingzoo.atari import basketball_pong_v3
            except:
                raise ImportError("pettingzoo[atari] is not installed! "
                                  "\nInstall it running: \npip install 'pettingzoo[atari]'")

            if 'num_players' not in kwargs.keys():
                kwargs['num_players'] = 2
            assert kwargs['num_players'] in [2, 4], "'num_players' should be 2 or 4!"

            return basketball_pong_v3.parallel_env(**kwargs)

        else:
            raise ValueError(f"Environment '{env_name}' is not supported.")

    def get_print_info(self):
        return self._env.print_info

    def step(self, actions):
        """ Returns reward, terminated, info """

        # Apply action for each agent
        actions = {f"{self.action_prefix[action_idx]}": action.item() if isinstance(action, np.int64) or
                                                                         (isinstance(action, np.ndarray) and
                                                                          str(action.dtype) == "int64")
                                                                      else
                                                         (action if isinstance(action, int)
                                                                 else
                                                          action.detach().cpu().item()
                                                          )
                   for action_idx, action in enumerate(actions)}

        self._obs, rewards, terminations, truncations, self._info = self._env.step(actions)

        if self.sum_rewards is True:
            # Add all rewards together
            reward = float(sum(rewards.values()))
        else:
            reward = list(rewards.values())

        done = (any([termination for termination in terminations.values()])
                or any([truncation for truncation in truncations.values()]))

        return reward, done, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """

        assert ((len(self.observation_space) == 1 and self.trainable_cnn is False) or
                (len(self.observation_space) == 3 and self.trainable_cnn is True)), \
               f"'self.observation_space': {self.observation_space}, 'self.trainable_cnn': {self.trainable_cnn}"
        return self.observation_space[0] if self.trainable_cnn is False else self.observation_space

    def get_state(self):

        if self.trainable_cnn is False and self.centralized_image_encoding is False:
            return np.concatenate(self._obs, axis=0).astype(np.float32)
        elif self.trainable_cnn is True and self.centralized_image_encoding is False:
            return np.stack(self._obs, axis=0).astype(np.float32)
        elif self.trainable_cnn is False and self.centralized_image_encoding is True:
            # In this case, the centralized encoder will encode observations and combine them to create the state
            return None
        else:
            raise NotImplementedError()

    def get_state_size(self):
        """ Returns the shape of the state"""

        assert ((len(self.observation_space) == 1 and self.trainable_cnn is False) or
                (len(self.observation_space) == 3 and self.trainable_cnn is True)), \
            f"'self.observation_space': {self.observation_space}, 'self.trainable_cnn': {self.trainable_cnn}"
        return self.n_agents * self.observation_space[0] if self.trainable_cnn is False \
                                                         else \
               (self.n_agents, *self.observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(self.action_prefix[agent_id])
            avail_actions.append(avail_agent)

        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return self._env.action_space(agent_id).n * [1]  # 1 indicates availability of actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.action_space

    def reset(self):
        """ Returns initial observations and states"""

        if self.key in ["entombed_cooperative_v3", "space_invaders_v2"]:

            # Here we fix the flickering issue of Atari 'entombed_cooperative_v3' and 'space_invaders_v2'
            # games when resetting the game.

            # Reset ony the original environment and get the obs
            previous_observations, previous_infos = self.original_env.reset()
            previous_obs = list(previous_observations.values())[0]

            # Perform no action in order to sync obs and actions
            no_actions = {'first_0': 0, 'second_0': 0}
            observations, rewards, terminations, truncations, infos = self.original_env.step(no_actions)
            current_obs = list(observations.values())[1]
            reward = sum(rewards.values())
            assert reward == 0, f"Reward greater than 0 found during resetting the game! Reward: {reward}"

            # Get the first combined obs of agents
            if self.key == "entombed_cooperative_v3":
                obs = self._env.entombed_cooperative_v3_get_combined_images(previous_obs, current_obs)
            elif self.key == "space_invaders_v2":
                obs = self._env.space_invaders_v2_get_combined_images(previous_obs, current_obs)
            else:
                raise ValueError(f"self.key: {self.key}")

            # Preprocess obs
            self._obs = self._env.observation(obs)

            # Simulate TimeLimit wrapper
            self._env.env._elapsed_steps = 1

        else:
            self._obs = self._env.reset()

        return self.get_obs(), self.get_state()

    def render(self):
        if self.render_mode != "human":  # otherwise it is already rendered
            # Get image
            env_image = self.original_env.render()
            # Convert RGB to BGR
            env_image = self.cv2.cvtColor(env_image, self.cv2.COLOR_RGB2BGR)
            # Render
            self.cv2.imshow(f"Environment: {self.key}", env_image)
            self.cv2.waitKey(1)

    def close(self):
        self._env.close()

    def seed(self):
        return self._seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
