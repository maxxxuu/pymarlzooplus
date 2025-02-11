import numpy as np
import torch
from gymnasium.utils import seeding

from smac.env import MultiAgentEnv
from pymarlzooplus.utils.image_encoder import ImageEncoder
from pymarlzooplus.utils.handle_import_errors import (
    import_error_pt_atari, import_error_pt_butterfly, import_error_pt_classic, import_error_pt_mpe, import_error_pt_sisl,
    atari_rom_error
)


class TimeLimitPZ(object):
    """Custom TimeLimit wrapper for PettingZoo environments."""
    def __init__(self, env, key, max_episode_steps):

        assert (
                isinstance(max_episode_steps, int) and max_episode_steps > 0
        ), f"Expect the `max_episode_steps` to be positive, actually: {max_episode_steps}"

        self.env = env
        self._key = key
        self._max_episode_steps = max_episode_steps

        # Placeholders
        self._elapsed_steps = None
        self._obs_wrapper = None

    def timelimit_step(self, actions):
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"

        infos = {}

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

                        rewards = {
                            reward_key: sum([reward, reward_, previous_reward, previous_reward_])
                            for (reward_key, reward, reward_, previous_reward, previous_reward_) in
                            zip(rewards.keys(), rewards.values(), rewards_.values(), previous_rewards.values(), previous_rewards_.values())
                        }

                        self._elapsed_steps += 4
                    else:  # Third step termination case
                        # In this case, we don't really care about observations
                        # since it's the last step (due to termination)
                        observations = previous_observations
                        rewards = {
                            reward_key: sum([reward_, previous_reward, previous_reward_])
                            for (reward_key, reward_, previous_reward, previous_reward_) in
                            zip(rewards_.keys(), rewards_.values(), previous_rewards.values(), previous_rewards_.values())
                        }
                        terminations = previous_terminations
                        truncations = previous_truncations
                        self._elapsed_steps += 3
                else:  # Second step termination case
                    # In this case, we don't really care about observations
                    # since it's the last step (due to termination)
                    observations = observations_
                    rewards = {
                        reward_key: sum([reward_, previous_reward_]) for (reward_key, reward_, previous_reward_) in
                        zip(rewards_.keys(), rewards_.values(), previous_rewards_.values())
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
            move_actions = {
                'first_0': actions['first_0'] if actions['first_0'] != 1 else 0,
                'second_0': actions['second_0'] if actions['second_0'] != 1 else 0
            }
            fire_actions = {
                'first_0': actions['first_0'] if actions['first_0'] == 1 else 0,
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

                rewards = {
                    reward_key: sum([reward, previous_reward]) for (reward_key, reward, previous_reward) in
                    zip(rewards.keys(), rewards.values(), previous_rewards.values())
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

        infos["TimeLimit.truncated"] = any([truncation for truncation in truncations.values()])

        if self._elapsed_steps >= self._max_episode_steps:
            terminations = {key: True for key in terminations.keys()}

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        self._elapsed_steps = 0
        return self.env.reset(seed=seed, options=options)

    def close(self):
        return self.env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space


class ObservationPZ(object):
    """
    Custom Observation wrapper for converting images to vectors (using a pretrained image encoder) or
    for preparing images to be fed to a CNN.
    """

    def __init__(
            self,
            env,
            partial_observation,
            trainable_cnn,
            image_encoder,
            image_encoder_batch_size,
            image_encoder_use_cuda,
            centralized_image_encoding,
            given_observation_space
    ):

        # Keep the parent environment
        self.env = env

        # Keep the original PettingZoo environment
        self.original_env = env.env

        # Keep the timelimit environment
        self.timelimit_env = env

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
        self.original_observation_space = self.original_env.observation_space(self.original_env.possible_agents[0])
        self.original_observation_space_shape = self.original_observation_space.shape
        self.is_image = len(self.original_observation_space_shape) == 3 and self.original_observation_space_shape[2] == 3
        assert self.is_image, f"Only images are supported, shape: {self.original_observation_space_shape}"

        self.given_observation_space = given_observation_space
        if given_observation_space is not None:
            self.observation_space = given_observation_space
        assert not (given_observation_space is None and centralized_image_encoding is True)
        assert not (given_observation_space is not None and centralized_image_encoding is False)

        # Placeholders
        self.cv2 = None

    def step(self, actions):
        if hasattr(self.timelimit_env, 'timelimit_step'):
            observations, rewards, terminations, truncations, infos = self.timelimit_env.timelimit_step(actions)
            return self.observation(observations), rewards, terminations, truncations, infos
        raise AttributeError("The 'timelimit_step' method is not implemented in the 'ObservationPZ'")

    def observation(self, observations):
        return self.image_encoder.observation(observations)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

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
        # Convert the difference to grayscale in order to handle a single threshold for all channels
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
        # Find which is the image that illustrates agent 1
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

    def close(self):
        return self.env.close()

    @property
    def action_space(self):
        return self.env.action_space


class _PettingZooWrapper(MultiAgentEnv):
    def __init__(
            self,
            key,
            time_limit=None,
            seed=1,
            render_mode="rgb_array",
            partial_observation=False,
            trainable_cnn=False,
            image_encoder="ResNet18",
            image_encoder_batch_size=1,
            image_encoder_use_cuda=False,
            centralized_image_encoding=False,
            kwargs="",
            given_observation_space=None
    ):

        assert (partial_observation is False) or (key in ["entombed_cooperative_v3", "space_invaders_v2"]), \
            (
                "'partial_observation' should be False when the selected game is other than "
                "'entombed_cooperative_v3' or 'space_invaders_v2'!"
            )

        self.key = key
        self.max_cycles = time_limit
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

        # Use cv2 for rendering
        import cv2
        self.cv2 = cv2

        # Placeholders
        self.kwargs = None
        self.original_env = None
        self.episode_limit = None
        self.n_agents = None
        self._env = None  # Observation wrapper
        self.__env = None  # TimeLimit wrapper
        self._obs = None
        self._info = None
        self.observation_space = None
        self.common_observation_space = None
        self.original_observation_space = None
        self.is_image = None
        self.action_space = None
        self.action_prefix = None
        self.sum_rewards = None
        self.np_random = None
        self.fully_cooperative_task_keys = [
            "pistonball_v6",
            "cooperative_pong_v5",
            "entombed_cooperative_v3",
            "space_invaders_v2"
        ]
        self.classic_task_keys = [
            "chess_v6",
            "connect_four_v3",
            "gin_rummy_v4",
            "go_v5",
            "hanabi_v5",
            "leduc_holdem_v4",
            "rps_v2",
            "texas_holdem_no_limit_v6",
            "texas_holdem_v4",
            "tictactoe_v3"
        ]

        # Environment
        self.set_environment(
            self.key,
            self.max_cycles,
            self.render_mode,
            self.partial_observation,
            self.trainable_cnn,
            self._image_encoder,
            self.image_encoder_batch_size,
            self.image_encoder_use_cuda,
            self.centralized_image_encoding,
            self._kwargs,
            self.given_observation_space
        )

    def set_environment(
            self,
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
            given_observation_space
    ):

        # Convert list of kwargs to dictionary
        self.kwargs = kwargs
        self.get_kwargs(max_cycles, render_mode)

        # Assign value to 'self.sum_rewards' based on the env key
        self.sum_rewards = True
        if key not in self.fully_cooperative_task_keys:
            # Only these environments refer to full cooperation, thus we can sum the rewards.
            self.sum_rewards = False

        # Define the environment
        self.original_env = self.pettingzoo_make(key, self.kwargs)

        # In the case of classic environments, return the original PettingZoo environment
        if self.key in self.classic_task_keys:
            return

        # Define episode horizon
        if hasattr(self.original_env.unwrapped, 'max_cycles'):
            self.episode_limit = self.original_env.unwrapped.max_cycles
        else:
            assert hasattr(self.original_env.unwrapped.env, 'max_cycles')
            self.episode_limit = self.original_env.unwrapped.env.max_cycles
            assert self.episode_limit > 1, "self.episode_limit should be > 1: {self.episode_limit}"

        # Define the number of agents
        self.n_agents = self.original_env.max_num_agents
        assert (self.n_agents == 2) or (key not in ["entombed_cooperative_v3", "space_invaders_v2"]), \
            (
                "Only 2 agents have been considered for 'entombed_cooperative_v3' and 'space_invaders_v2'! "
                "'n_agents': {}".format(self.n_agents)
            )

        # Define TimeLimit wrapper
        self.__env = TimeLimitPZ(
            self.original_env,
            key,
            max_episode_steps=self.episode_limit
        )
        # In the case of "entombed_cooperative_v3" and "space_invaders_v2" games,
        # fix the "self.episode_limit" after passing it to the "TimeLimitPZ"
        if key == "entombed_cooperative_v3":
            # At each timestep, we apply 4 pettingzoo timesteps,
            # in order to synchronize actions and obs
            self.episode_limit = int(self.episode_limit / 4)
        elif key == "space_invaders_v2":
            # At each timestep, we apply 2 pettingzoo timesteps,
            # in order to synchronize actions and obs
            self.episode_limit = int(self.episode_limit / 2)

        # Define Observation wrapper
        self.original_observation_space = self.__env.observation_space(self.original_env.possible_agents[0])
        self.common_observation_space = all(
            [
                self.original_observation_space == self.__env.observation_space(self.original_env.possible_agents[agent_id])
                for agent_id in range(self.n_agents)
            ]
        )
        self.is_image = len(self.original_observation_space.shape) == 3 and self.original_observation_space.shape[2] == 3
        if self.is_image is True:
            self._env = ObservationPZ(
                self.__env,
                partial_observation,
                trainable_cnn,
                image_encoder,
                image_encoder_batch_size,
                image_encoder_use_cuda,
                centralized_image_encoding,
                given_observation_space
            )
            self._env.cv2 = self.cv2
            self.__env._obs_wrapper = self._env
        else:
            # Just for consistency
            if self.common_observation_space is False:
                self.original_observation_space = self.original_env.observation_spaces  # dictionary
            self._env = self.__env
            setattr(self._env, 'observation_space', self.original_observation_space)

        self._obs = None
        self._info = None

        # Define the observation space
        self.observation_space = self._env.observation_space

        ## Define the action space
        tmp_action_spaces_list = [
            self.original_env.action_space(possible_agent) for possible_agent in self.original_env.possible_agents
        ]
        tmp_action_space = tmp_action_spaces_list[0]
        if key in self.fully_cooperative_task_keys:
            # Check that all agents have the same action space only for fully cooperation tasks
            assert not isinstance(tmp_action_space, tuple), f"tmp_action_space: {tmp_action_space}"
            for agent_idx, tmp_action_space_ in enumerate(tmp_action_spaces_list[1:], start=1):
                assert tmp_action_space_ == tmp_action_space, (
                    "Difference in action spaces found between agents:\n"
                     f"agent=0 - action_space={tmp_action_space}, agent={agent_idx} - action_space={tmp_action_space_}"
                )
            self.action_space = tmp_action_space.n.item()
        else:
            self.action_space = tmp_action_space
        self.action_prefix = [action_prefix for action_prefix in self.original_env.possible_agents]

        # Create the seed object to control the randomness of the environment reset()
        self.np_random, self._seed = seeding.np_random(self._seed)

    def get_kwargs(self, max_cycles, render_mode):
        if (
                isinstance(self.kwargs, (list, tuple)) or
                (isinstance(self.kwargs, str) and len(self.kwargs) > 0 and isinstance(eval(self.kwargs), tuple))
        ):
            if not isinstance(self.kwargs, (list, tuple)) and isinstance(eval(self.kwargs), (tuple, list)):
                self.kwargs = eval(self.kwargs)
            if isinstance(self.kwargs[0], (list, tuple)):
                # Convert the list to dict
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

        #### Butterfly environments ####

        if env_name == "pistonball_v6":

            if 'continuous' not in kwargs.keys():
                kwargs['continuous'] = False
            assert kwargs['continuous'] is False, "'continuous' argument should be False!"
            if 'n_pistons' in kwargs.keys():
                assert kwargs['n_pistons'] >= 4, \
                    "'n_pistons' argument must be greater than or equal to 4!"
                # Otherwise, the game stops almost immediately.

            try:

                from pettingzoo.butterfly import pistonball_v6
                return pistonball_v6.parallel_env(**kwargs)  # Parallel mode

            except ImportError:
                import_error_pt_butterfly()

        elif env_name == "cooperative_pong_v5":
            try:

                from pettingzoo.butterfly import cooperative_pong_v5
                return cooperative_pong_v5.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_butterfly()

        elif env_name == "knights_archers_zombies_v10":

            if 'vector_state' not in kwargs.keys():
                kwargs['vector_state'] = False
            assert kwargs['vector_state'] is False, "'vector_state' argument should be False!"
            if 'use_typemasks' not in kwargs.keys():
                kwargs['use_typemasks'] = False
            assert kwargs['use_typemasks'] is False, "'use_typemasks' argument should be False!"
            if 'sequence_space' not in kwargs.keys():
                kwargs['sequence_space'] = False
            assert kwargs['sequence_space'] is False, "'sequence_space' argument should be False!"

            try:

                from pettingzoo.butterfly import knights_archers_zombies_v10
                return knights_archers_zombies_v10.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_butterfly()

        #### Atari environments ####

        elif env_name == "entombed_cooperative_v3":

            try:

                from pettingzoo.atari import entombed_cooperative_v3
                return entombed_cooperative_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "space_invaders_v2":

            if 'alternating_control' not in kwargs.keys():
                kwargs['alternating_control'] = False
            assert kwargs['alternating_control'] is False, "'alternating_control' should be False!"

            try:

                from pettingzoo.atari import space_invaders_v2
                return space_invaders_v2.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "basketball_pong_v3":

            if 'num_players' not in kwargs.keys():
                kwargs['num_players'] = 2
            assert kwargs['num_players'] in [2, 4], "'num_players' should be 2 or 4!"

            try:

                from pettingzoo.atari import basketball_pong_v3
                return basketball_pong_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "boxing_v2":

            try:

                from pettingzoo.atari import boxing_v2
                return boxing_v2.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "combat_plane_v2":

            if 'game_version' in kwargs.keys():
                assert kwargs['game_version'] in ["jet", "bi-plane"], \
                    "'game_version' should be one of following: ['jet', 'bi-plane']!"

            try:
                # There is an inconsistency in PettingZoo Documentation,
                # they say to import like this
                # (top of the page: https://pettingzoo.farama.org/environments/atari/combat_plane/):
                # from pettingzoo.atari import combat_jet_v1
                # then, they use (https://pettingzoo.farama.org/environments/atari/combat_plane/#environment-parameters):
                # combat_plane_v2.env(game_version="jet", guided_missile=True)
                from pettingzoo.atari import combat_plane_v2
                return combat_plane_v2.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "combat_tank_v2":
            try:
                # There is an inconsistency in PettingZoo Documentation,
                # they say to import like this
                # (top of the page: https://pettingzoo.farama.org/environments/atari/combat_tank/):
                # from pettingzoo.atari import combat_tank_v3
                # then, they use (https://pettingzoo.farama.org/environments/atari/combat_tank/#environment-parameters):
                # combat_tank_v2.env(has_maze=True, is_invisible=False, billiard_hit=True)
                from pettingzoo.atari import combat_tank_v2
                return combat_tank_v2.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "double_dunk_v3":
            try:

                from pettingzoo.atari import double_dunk_v3
                return double_dunk_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "entombed_competitive_v3":
            try:

                from pettingzoo.atari import entombed_competitive_v3
                return entombed_competitive_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "flag_capture_v2":
            try:

                from pettingzoo.atari import flag_capture_v2
                return flag_capture_v2.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "foozpong_v3":

            if 'num_players' not in kwargs.keys():
                kwargs['num_players'] = 2
            assert kwargs['num_players'] in [2, 4], "'num_players' should be 2 or 4!"

            try:

                from pettingzoo.atari import foozpong_v3
                return foozpong_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "ice_hockey_v2":
            try:

                from pettingzoo.atari import ice_hockey_v2
                return ice_hockey_v2.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "joust_v3":
            try:

                from pettingzoo.atari import joust_v3
                return joust_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "mario_bros_v3":
            try:

                from pettingzoo.atari import mario_bros_v3
                return mario_bros_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "maze_craze_v3":

            if 'game_version' in kwargs.keys():
                assert kwargs['game_version'] in ["robbers", "race", "capture"], \
                    "'game_version' should be one of following: ['robbers', 'race', 'capture']!"
            if 'visibilty_level' in kwargs.keys():
                assert kwargs['visibilty_level'] in [0, 1, 2, 3], \
                    "'visibilty_level' should be one of following: [0, 1, 2, 3]!"

            try:

                from pettingzoo.atari import maze_craze_v3
                return maze_craze_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "othello_v3":
            try:

                from pettingzoo.atari import othello_v3
                return othello_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "pong_v3":

            if 'num_players' not in kwargs.keys():
                kwargs['num_players'] = 2
            assert kwargs['num_players'] in [2, 4], "'num_players' should be 2 or 4!"

            try:

                from pettingzoo.atari import pong_v3
                return pong_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "quadrapong_v4":
            try:

                from pettingzoo.atari import quadrapong_v4
                return quadrapong_v4.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "space_war_v2":
            try:

                from pettingzoo.atari import space_war_v2
                return space_war_v2.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "surround_v2":
            try:

                from pettingzoo.atari import surround_v2
                return surround_v2.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "tennis_v3":
            try:

                from pettingzoo.atari import tennis_v3
                return tennis_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "video_checkers_v4":
            try:

                from pettingzoo.atari import video_checkers_v4
                return video_checkers_v4.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "volleyball_pong_v3":

            if 'num_players' not in kwargs.keys():
                kwargs['num_players'] = 4
            assert kwargs['num_players'] in [2, 4], "'num_players' should be 2 or 4!"

            try:
                # There is an inconsistency in PettingZoo Documentation,
                # they say to import like this
                # (top of the page: https://pettingzoo.farama.org/environments/atari/volleyball_pong/):
                # from pettingzoo.atari import volleyball_pong_v2
                # then, they use (https://pettingzoo.farama.org/environments/atari/volleyball_pong/#environment-parameters):
                # volleyball_pong_v3.env(num_players=4)
                from pettingzoo.atari import volleyball_pong_v3
                return volleyball_pong_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "warlords_v3":
            try:

                from pettingzoo.atari import warlords_v3
                return warlords_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        elif env_name == "wizard_of_wor_v3":
            try:

                from pettingzoo.atari import wizard_of_wor_v3
                return wizard_of_wor_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_atari()

            except OSError as e:
                atari_rom_error(e)

        #### Classic environments ####

        elif env_name == "chess_v6":

            if 'max_cycles' in kwargs.keys():
                del kwargs['max_cycles']

            try:
                from pettingzoo.classic import chess_v6
                return chess_v6.env(**kwargs)

            except ImportError:
                import_error_pt_classic()

        elif env_name == "connect_four_v3":

            if 'max_cycles' in kwargs.keys():
                del kwargs['max_cycles']

            try:
                from pettingzoo.classic import connect_four_v3
                return connect_four_v3.env(**kwargs)

            except ImportError:
                import_error_pt_classic()

        elif env_name == "gin_rummy_v4":

            if 'max_cycles' in kwargs.keys():
                del kwargs['max_cycles']

            try:
                from pettingzoo.classic import gin_rummy_v4
                return gin_rummy_v4.env(**kwargs)

            except ImportError:
                import_error_pt_classic()

        elif env_name == "go_v5":

            if 'max_cycles' in kwargs.keys():
                del kwargs['max_cycles']

            try:
                from pettingzoo.classic import go_v5
                return go_v5.env(**kwargs)

            except ImportError:
                import_error_pt_classic()

        elif env_name == "hanabi_v5":

            if 'max_cycles' in kwargs.keys():
                del kwargs['max_cycles']
            if 'players' not in kwargs.keys():
                kwargs['players'] = 2
            if 'hand_size' not in kwargs.keys():
                if kwargs['players'] >= 4:
                    kwargs['hand_size'] = 4
                else:
                    kwargs['hand_size'] = 5
            else:
                if kwargs['players'] >= 4:
                    assert kwargs['hand_size'] == 4, "When 'players'>=4, 'hand_size' should be 4!"
                else:
                    assert kwargs['hand_size'] == 5, "When 'players'<4, 'hand_size' should be 5!"

            try:
                from pettingzoo.classic import hanabi_v5
                return hanabi_v5.env(**kwargs)

            except ImportError:
                import_error_pt_classic()

        elif env_name == "leduc_holdem_v4":

            if 'max_cycles' in kwargs.keys():
                del kwargs['max_cycles']

            try:
                from pettingzoo.classic import leduc_holdem_v4
                return leduc_holdem_v4.env(**kwargs)

            except ImportError:
                import_error_pt_classic()

        elif env_name == "rps_v2":

            if 'num_actions' not in kwargs.keys():
                kwargs['num_actions'] = 3
            assert kwargs['num_actions'] in [3, 5], "'num_actions' should be either 3 or 5!"

            try:
                from pettingzoo.classic import rps_v2
                return rps_v2.env(**kwargs)

            except ImportError:
                import_error_pt_classic()

        elif env_name == "texas_holdem_no_limit_v6":

            if 'max_cycles' in kwargs.keys():
                del kwargs['max_cycles']
            if 'num_players' not in kwargs.keys():
                kwargs['num_players'] = 2
            assert kwargs['num_players'] >= 2, "'num_players' should be 2 or greater!"

            try:
                from pettingzoo.classic import texas_holdem_no_limit_v6
                return texas_holdem_no_limit_v6.env(**kwargs)

            except ImportError:
                import_error_pt_classic()

        elif env_name == "texas_holdem_v4":

            if 'max_cycles' in kwargs.keys():
                del kwargs['max_cycles']
            if 'num_players' not in kwargs.keys():
                kwargs['num_players'] = 2
            assert kwargs['num_players'] >= 2, "'num_players' should be 2 or greater!"

            try:
                from pettingzoo.classic import texas_holdem_v4
                return texas_holdem_v4.env(**kwargs)

            except ImportError:
                import_error_pt_classic()

        elif env_name == "tictactoe_v3":

            if 'max_cycles' in kwargs.keys():
                del kwargs['max_cycles']

            try:
                from pettingzoo.classic import tictactoe_v3
                return tictactoe_v3.env(**kwargs)

            except ImportError:
                import_error_pt_classic()

        #### MPE environments ####

        elif env_name == "simple_v3":
            try:

                from pettingzoo.mpe import simple_v3
                return simple_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_mpe()

        elif env_name == "simple_adversary_v3":

            if 'N' not in kwargs.keys():
                kwargs['N'] = 2
            assert kwargs['N'] >= 2, "'N' should be 2 or greater!"

            try:

                from pettingzoo.mpe import simple_adversary_v3
                return simple_adversary_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_mpe()

        elif env_name == "simple_crypto_v3":
            try:

                from pettingzoo.mpe import simple_crypto_v3
                return simple_crypto_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_mpe()

        elif env_name == "simple_push_v3":
            try:

                from pettingzoo.mpe import simple_push_v3
                return simple_push_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_mpe()

        elif env_name == "simple_reference_v3":
            try:

                from pettingzoo.mpe import simple_reference_v3
                return simple_reference_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_mpe()

        elif env_name == "simple_speaker_listener_v4":
            try:

                from pettingzoo.mpe import simple_speaker_listener_v4
                return simple_speaker_listener_v4.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_mpe()

        elif env_name == "simple_spread_v3":

            if 'N' not in kwargs.keys():
                kwargs['N'] = 3
            assert kwargs['N'] >= 3, "'N' should be 2 or greater!"

            try:

                from pettingzoo.mpe import simple_spread_v3
                return simple_spread_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_mpe()

        elif env_name == "simple_tag_v3":

            if 'num_good' not in kwargs.keys():
                kwargs['num_good'] = 1
            assert kwargs['num_good'] >= 0, "'num_good' should be 0 or greater!"
            if 'num_adversaries' not in kwargs.keys():
                kwargs['num_adversaries'] = 3
            assert kwargs['num_adversaries'] >= 0, "'num_adversaries' should be 0 or greater!"
            if 'num_obstacles' not in kwargs.keys():
                kwargs['num_obstacles'] = 2
            assert kwargs['num_obstacles'] >= 0, "'num_obstacles' should be 0 or greater!"

            try:

                from pettingzoo.mpe import simple_tag_v3
                return simple_tag_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_mpe()

        elif env_name == "simple_world_comm_v3":

            if 'num_good' not in kwargs.keys():
                kwargs['num_good'] = 2
            assert kwargs['num_good'] >= 0, "'num_good' should be 0 or greater!"
            if 'num_adversaries' not in kwargs.keys():
                kwargs['num_adversaries'] = 4
            assert kwargs['num_adversaries'] >= 0, "'num_adversaries' should be 0 or greater!"
            if 'num_obstacles' not in kwargs.keys():
                kwargs['num_obstacles'] = 1
            assert kwargs['num_obstacles'] >= 0, "'num_obstacles' should be 0 or greater!"
            if 'num_food' not in kwargs.keys():
                kwargs['num_food'] = 1
            assert kwargs['num_food'] >= 0, "'num_food' should be 0 or greater!"
            if 'num_forests' not in kwargs.keys():
                kwargs['num_forests'] = 1
            assert kwargs['num_forests'] >= 0, "'num_forests' should be 0 or greater!"

            try:

                from pettingzoo.mpe import simple_world_comm_v3
                return simple_world_comm_v3.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_mpe()

        #### SISL environments ####

        elif env_name == "multiwalker_v9":

            if 'n_walkers' not in kwargs.keys():
                kwargs['n_walkers'] = 3
            assert kwargs['n_walkers'] >= 2, "'n_walkers' should be 2 or greater!"
            if 'terrain_length' not in kwargs.keys():
                kwargs['terrain_length'] = 200
            assert kwargs['terrain_length'] >= 50, "'terrain_length' should be 50 or greater!"

            try:

                from pettingzoo.sisl import multiwalker_v9
                return multiwalker_v9.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_sisl()

        elif env_name == "pursuit_v4":

            if 'x_size' not in kwargs.keys():
                kwargs['x_size'] = 16
            assert kwargs['x_size'] >= 2, "'x_size' should be 2 or greater!"
            if 'y_size' not in kwargs.keys():
                kwargs['y_size'] = 16
            assert kwargs['y_size'] >= 2, "'y_size' should be 2 or greater!"
            if 'n_evaders' not in kwargs.keys():
                kwargs['n_evaders'] = 30
            assert kwargs['n_evaders'] >= 1, "'n_evaders' should be 1 or greater!"
            if 'n_pursuers' not in kwargs.keys():
                kwargs['n_pursuers'] = 8
            assert kwargs['n_pursuers'] >= 1, "'n_pursuers' should be 1 or greater!"
            if 'obs_range' not in kwargs.keys():
                kwargs['obs_range'] = 7
            assert kwargs['obs_range'] >= 1, "'obs_range' should be 1 or greater!"
            if 'n_catch' not in kwargs.keys():
                kwargs['n_catch'] = 2
            assert kwargs['n_catch'] >= 1, "'n_catch' should be 1 or greater!"

            try:

                from pettingzoo.sisl import pursuit_v4
                return pursuit_v4.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_sisl()

        elif env_name == "waterworld_v4":

            if 'n_evaders' not in kwargs.keys():
                kwargs['n_evaders'] = 5
            assert kwargs['n_evaders'] >= 1, "'n_evaders' should be 1 or greater!"
            if 'n_pursuers' not in kwargs.keys():
                kwargs['n_pursuers'] = 5
            assert kwargs['n_pursuers'] >= 1, "'n_pursuers' should be 1 or greater!"
            if 'n_poisons' not in kwargs.keys():
                kwargs['n_poisons'] = 10
            assert kwargs['n_poisons'] >= 0, "'n_poisons' should be 0 or greater!"
            if 'n_coop' not in kwargs.keys():
                kwargs['n_coop'] = 2
            assert kwargs['n_coop'] >= 0, "'n_coop' should be 0 or greater!"
            if 'n_sensors' not in kwargs.keys():
                kwargs['n_sensors'] = 20
            assert kwargs['n_sensors'] >= 1, "'n_sensors' should be 1 or greater!"
            assert 'obstacle_coord' not in kwargs.keys(), "'obstacle_coord' specification is not supported yet!"

            try:

                from pettingzoo.sisl import waterworld_v4
                return waterworld_v4.parallel_env(**kwargs)

            except ImportError:
                import_error_pt_sisl()

        else:
            raise ValueError(f"Environment '{env_name}' is not supported.")

    def get_print_info(self):
        return self._env.print_info

    def step(self, actions):
        """ Returns reward, terminated, info """

        # Fix the actions' type
        fixed_actions = {}
        for action_idx, action in enumerate(actions):
            if (
                    isinstance(action, (np.int64, np.int32, np.float64, np.float32)) or
                    (isinstance(action, np.ndarray) and str(action.dtype) in ["int64", "int32", "float64", "float32"])
            ):
                if len(action.flatten()) == 1:
                    tmp_action = action.item()
                else:
                    assert len(action.shape) == 1, f"len(action.shape): {len(action.shape)}"
                    tmp_action = action
            elif isinstance(action, (int, float)):
                tmp_action = action
            elif isinstance(action, list):
                tmp_action = np.array(action)
            elif isinstance(action, torch.Tensor):
                tmp_action = action.detach().cpu().item()
            else:
                raise NotImplementedError(f"Not supported action type! type(action): {type(action)}")

            fixed_actions[self.action_prefix[action_idx]] = tmp_action

        # Apply action for each agent
        self._obs, rewards, terminations, truncations, self._info = self._env.step(fixed_actions)

        if self.sum_rewards is True: # The case of fully cooperative tasks
            # Add all rewards together
            reward = float(sum(rewards.values()))
            # 'done' is True if there is at least a single truncation or termination
            done = (
                    any([termination for termination in terminations.values()]) or
                    any([truncation for truncation in truncations.values()])
            )
            # Keep only 'TimeLimit.truncated' in 'self._info'
            self._info = {'TimeLimit.truncated': self._info['TimeLimit.truncated']}
            return reward, done, {}
        else: # The case of NOT fully cooperative tasks
            timelimit_truncated = self._info['TimeLimit.truncated']
            del self._info['TimeLimit.truncated']
            return (
                rewards,
                terminations,
                {'truncations': truncations, 'infos': self._info, 'TimeLimit.truncated': timelimit_truncated}
            )


    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """

        # Image observations
        if self.is_image is True:
            assert (
                    (len(self.observation_space) == 1 and self.trainable_cnn is False) or
                    (len(self.observation_space) == 3 and self.trainable_cnn is True)
            ), f"'self.observation_space': {self.observation_space}, 'self.trainable_cnn': {self.trainable_cnn}"
            return self.observation_space[0] if self.trainable_cnn is False else self.observation_space

        # Vector observations
        else:
            if self.common_observation_space is False:
                return self.observation_space
            else:
                if (
                        (isinstance(self._obs, tuple) and len(self._obs) == 2) or
                        (isinstance(self._obs, dict) and len(self._obs) == self.n_agents)
                ):
                    if isinstance(self._obs, tuple):
                        assert all([self._obs[1][_action_prefix] == {} for _action_prefix in self.action_prefix]), \
                            f"self._obs: {self._obs}"
                    return self.observation_space.shape
                else:
                    raise NotImplementedError

    def get_state(self):

        # Image observations
        if self.is_image is True:
            if self.trainable_cnn is False and self.centralized_image_encoding is False:
                return np.concatenate(self._obs, axis=0).astype(np.float32)
            elif self.trainable_cnn is True and self.centralized_image_encoding is False:
                return np.stack(self._obs, axis=0).astype(np.float32)
            elif self.trainable_cnn is False and self.centralized_image_encoding is True:
                # In this case, the centralized encoder will encode observations and combine them to create the state
                return None
            else:
                raise NotImplementedError()

        # Vector observations
        else:
            if self.common_observation_space is False:
                # Return observations as given by PettingZoo
                return self._obs
            else:
                if (
                        (isinstance(self._obs, tuple) and len(self._obs) == 2) or
                        (isinstance(self._obs, dict) and len(self._obs) == self.n_agents)
                ):
                    if isinstance(self._obs, tuple):
                        assert all([self._obs[1][_action_prefix] == {} for _action_prefix in self.action_prefix]), \
                            f"self._obs: {self._obs}"
                        self._obs = self._obs[0]
                    self._obs = [self._obs[_action_prefix] for _action_prefix in self.action_prefix]
                    return np.stack(self._obs, axis=0).astype(np.float32)
                else:
                    raise NotImplementedError


    def get_state_size(self):
        """ Returns the shape of the state"""

        # Image observations
        if self.is_image is True:
            assert (
                    (len(self.observation_space) == 1 and self.trainable_cnn is False) or
                    (len(self.observation_space) == 3 and self.trainable_cnn is True)
            ), f"'self.observation_space': {self.observation_space}, 'self.trainable_cnn': {self.trainable_cnn}"
            return self.n_agents * self.observation_space[0] if self.trainable_cnn is False \
                                                             else \
                   (self.n_agents, *self.observation_space)

        # Vector observations
        else:
            if self.common_observation_space is False:
                return None
            else:
                if (
                        (isinstance(self._obs, tuple) and len(self._obs) == 2) or
                        (isinstance(self._obs, dict) and len(self._obs) == self.n_agents)
                ):
                    if isinstance(self._obs, tuple):
                        assert all([self._obs[1][_action_prefix] == {} for _action_prefix in self.action_prefix]), \
                            f"self._obs: {self._obs}"
                    return tuple((self.n_agents, *self.observation_space.shape))
                else:
                    raise NotImplementedError

    def get_avail_actions(self):

        if isinstance(self.action_space, int):
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(self.action_prefix[agent_id])
                avail_actions.append(avail_agent)

            return avail_actions
        else:
            raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        if isinstance(self.action_space, int):
            return self._env.action_space(agent_id).n * [1]  # 1 indicates availability of actions
        else:
            raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        if isinstance(self.action_space, int):
            return self.action_space
        else:
            raise NotImplementedError

    def sample_actions(self):
        return [self._env.action_space(agent).sample() for agent in self._env.agents]

    def reset(self, seed=None):
        """ Returns initial observations and states"""

        # Control seed
        if seed is None:
            self._seed = self.np_random.choice(np.iinfo(np.int32).max)
        else:
            self.np_random, self._seed = seeding.np_random(self._seed)

        if self.key in ["entombed_cooperative_v3", "space_invaders_v2"]:

            # Here we fix the flickering issue of Atari 'entombed_cooperative_v3' and 'space_invaders_v2'
            # games when resetting the game.

            # Reset only the original environment and get the obs
            previous_observations, previous_infos = self.original_env.reset(seed=self._seed)
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
            self._obs, _ = self._env.reset(seed=self._seed)

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

    @staticmethod
    def get_stats():
        return {}
