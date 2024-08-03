from functools import partial
import sys
import os

from smac.env import MultiAgentEnv, StarCraft2Env

import numpy as np
import gym
from gym import ObservationWrapper, spaces
from gym.spaces import flatdim
from gym.wrappers import TimeLimit as GymTimeLimit
from gym.envs.registration import register


############################################
# Registration of SMAC

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {"sc2": partial(env_fn, env=StarCraft2Env)}

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )
############################################


############################################
# Registration of gym

class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done) \
                if type(done) is list \
                else not done
            done = len(observation) * [True]
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim_ = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(low=-float("inf"),
                           high=float("inf"),
                           shape=(flatdim_,),
                           dtype=np.float32,)
                         ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):

        return tuple(
            [spaces.flatten(obs_space, obs) for obs_space, obs in zip(self.env.observation_space, observation)]
                    )


class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, seed, **kwargs):
        self.original_env = gym.make(f"{key}", **kwargs)
        self.episode_limit = time_limit
        self._env = TimeLimit(self.original_env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        self.n_agents = self._env.n_agents
        self._obs = None
        self._info = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(self._env.observation_space, key=lambda x: x.shape)

        self._seed = seed
        self._env.seed(self._seed)

    def step(self, actions):
        """ Returns reward, terminated, info """

        actions = [int(a) for a in actions]
        self._obs, reward, done, self._info = self._env.step(actions)
        self._obs = [np.pad(o,
                            (0, self.longest_observation_space.shape[0] - len(o)),
                            "constant",
                            constant_values=0,)
                     for o in self._obs]

        if type(reward) is list:
            reward = sum(reward)
        if type(done) is list:
            done = all(done)

        return float(reward), done, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """

        return flatdim(self.longest_observation_space)

    def get_state(self):

        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        if hasattr(self.original_env, 'state_size'):
            return self.original_env.state_size

        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)

        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))

        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent

        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        self._obs = [np.pad(o,
                            (0, self.longest_observation_space.shape[0] - len(o)),
                            "constant",
                            constant_values=0,)
                     for o in self._obs]

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


REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
############################################

############################################
# Registration for PettingZoo


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
                 image_encoder_use_cuda):

        super(ObservationPZ, self).__init__(env)

        self.print_info = None
        self.partial_observation = partial_observation
        self.trainable_cnn = trainable_cnn
        self.original_observation_space = self.env.observation_space(self.env.possible_agents[0])
        self.original_observation_space_shape = self.original_observation_space.shape
        self.is_image = len(self.original_observation_space_shape) == 3 and self.original_observation_space_shape[2] == 3
        assert self.is_image, f"Only images are supported, shape: {self.original_observation_space_shape}"

        # Import pettingzoo specific requirements
        import torch
        self.torch = torch
        import cv2
        self.cv2 = cv2

        ## Define image encoder. In this case, a pretrained model is used frozen, i.e., without further training.
        self.image_encoder = None
        if self.is_image and self.trainable_cnn is False:

            # Define the device to be used
            self.device = "cpu"
            if image_encoder_use_cuda is True and self.torch.cuda.is_available() is True:
                self.device = "cuda"

            # Define the batch size of the image encoder
            self.image_encoder_batch_size = image_encoder_batch_size

            # Encoder
            self.image_encoder = None
            self.image_encoder_predict = None
            if image_encoder == "ResNet18":

                # Imports
                import albumentations as A
                from albumentations.pytorch import ToTensorV2
                from torch import nn
                from torchvision.models import resnet18

                # Define ResNet18
                self.print_info = "Loading pretrained ResNet18 model..."
                self.image_encoder = resnet18(weights='IMAGENET1K_V1')
                self.image_encoder.fc = nn.Identity()
                self.image_encoder = self.image_encoder.to(self.device)
                self.image_encoder.eval()

                # Image transformations
                img_size = 224
                self.transform = A.Compose([
                    A.LongestMaxSize(max_size=img_size, interpolation=1), # Resize the longest side to 224
                    A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=(0, 0, 0)), # Pad to make the image square
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                                            ])

                # Get the number of features by feeding the model with a dummy input
                dummy_input = np.ones((1, img_size, img_size, 3), dtype=np.uint8)*255
                dummy_output = self.resnet18_predict(dummy_input)
                n_features = dummy_output.shape[1]

                # Define the function to get predictions
                self.image_encoder_predict = self.resnet18_predict

            elif image_encoder == "SlimSAM":

                # Imports
                from transformers import SamModel, SamProcessor

                # Define SAM
                self.print_info = "Loading pretrained SlimSAM model..."
                self.image_encoder = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50").to(self.device) # Original SAM: facebook/sam-vit-base, options: huge, large, base
                self.image_encoder.eval()

                # Image transformations
                self.processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")

                # Get the number of features by feeding the model with a dummy input
                img_size = 224
                dummy_input = np.ones((1, img_size, img_size, 3), dtype=np.uint8) * 255
                dummy_output = self.sam_predict(dummy_input)
                n_features = dummy_output.shape[1]

                # Define the function to get predictions
                self.image_encoder_predict = self.resnet18_predict

            elif image_encoder == "CLIP":

                # Imports
                from transformers import AutoProcessor, CLIPVisionModel

                # Define CLIP-image-encoder
                self.print_info = "Loading pretrained CLIP-image-encoder model..."
                self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                self.image_encoder.eval()

                # Image transformations
                self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

                # Get the number of features by feeding the model with a dummy input
                img_size = 224
                dummy_input = np.ones((1, img_size, img_size, 3), dtype=np.uint8) * 255
                dummy_output = self.clip_predict(dummy_input)
                n_features = dummy_output.shape[1]

                # Define the function to get predictions
                self.image_encoder_predict = self.clip_predict

            else:
                raise NotImplementedError(f"Invalid image encoder: {image_encoder}")

            # Define the observation space
            self.observation_space = (n_features,)

        elif self.is_image and self.trainable_cnn is True:
            # In this case, we use NatureCNN, adopted from openAI:
            # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py

            # Image transformations.
            # Images will be downscaled to 224 max size, reducing the complexity,
            # and will be padded (if needed) to become square.
            # Images will be normalized simply by dividing by 255 (as in the original Nature paper,
            # but without converting to gray scale).
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            img_size = 224
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=img_size, interpolation=1),  # Resize the longest side to 224
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=(0, 0, 0)),  # Pad to make the image square
                ToTensorV2()
            ])

            # Define the observation space
            self.observation_space = (3, img_size, img_size)
        else:
            raise NotImplementedError("Only images are supported!")

    def resnet18_predict(self, observation):
        """
        observation: np.array of shape [batch_size, height, width, 3]
        """

        assert isinstance(observation, np.ndarray), \
            f"'observation' is not a numpy array! 'type(observation)': {type(observation)} "
        assert observation.ndim == 4 and observation.shape[3] == 3, \
            f"'observation' has not the right dimensions! 'observation.shape': {observation.shape}"

        observation = [self.transform(image=obs)["image"][None] for obs in observation]
        observation = self.torch.concatenate(observation, dim=0)
        observation = observation.to(self.device)
        with self.torch.no_grad():
            observation = self.image_encoder(observation)
            observation = observation.detach().cpu().numpy()

        return observation

    def sam_predict(self, observation):
        """
        observation: np.array of shape [batch_size, height, width, 3]
        """

        assert isinstance(observation, np.ndarray), \
            f"'observation' is not a numpy array! 'type(observation)': {type(observation)} "
        assert observation.ndim == 4 and observation.shape[3] == 3, \
            f"'observation' has not the right dimensions! 'observation.shape': {observation.shape}"

        observation = self.processor(observation, return_tensors="pt")['pixel_values'].to(self.device)
        with self.torch.no_grad():
            observation = self.image_encoder.get_image_embeddings(pixel_values=observation)
            bs = observation.shape[0]
            observation = observation.view((bs, -1)).detach().cpu().numpy()

        return observation

    def clip_predict(self, observation):
        """
        observation: np.array of shape [batch_size, height, width, 3]
        """

        assert isinstance(observation, np.ndarray), \
            f"'observation' is not a numpy array! 'type(observation)': {type(observation)} "
        assert observation.ndim == 4 and observation.shape[3] == 3, \
            f"'observation' has not the right dimensions! 'observation.shape': {observation.shape}"

        observation = self.processor(images=observation, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            observation = self.image_encoder(**observation).pooler_output
            observation = observation.detach().cpu().numpy()

        return observation

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return self.observation(observations), rewards, terminations, truncations, infos

    def observation(self, observations):

        if isinstance(observations, tuple):
            # When 'observations' is tuple it means that it has been called from gym reset()
            # and it carries pettingzoo observations and info
            observations = observations[0]

        observations_ = []
        if self.is_image and self.trainable_cnn is False:
            # Get image representations
            observations_tmp = []
            observations_tmp_counter = 0
            observations_tmp_counter_total = 0
            for observation_ in observations.values():
                observations_tmp.append(observation_[None])
                observations_tmp_counter += 1
                observations_tmp_counter_total += 1
                if observations_tmp_counter == self.image_encoder_batch_size or \
                   observations_tmp_counter_total == len(observations.values()):
                    # Predict in batches. When GPU is used, this is faster than inference over single images.
                    observations_tmp = np.concatenate(observations_tmp, axis=0)
                    observations_tmp = self.image_encoder_predict(observations_tmp)
                    observations_.extend([obs for obs in observations_tmp])
                    # Reset tmp
                    observations_tmp = []
                    observations_tmp_counter = 0
        elif self.is_image and self.trainable_cnn is True:
            # Preprocess images for a CNN network.
            observations_ = [self.transform(image=observation_)["image"].detach().cpu().numpy()
                             for observation_ in observations.values()]
        else:
            raise NotImplementedError("Only images are supported!")

        return tuple(observations_,)

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
                 render_mode,
                 partial_observation,
                 trainable_cnn,
                 image_encoder,
                 image_encoder_batch_size,
                 image_encoder_use_cuda,
                 kwargs):

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
        self._kwargs = kwargs

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

        # Environment
        self.set_environment(self.key,
                             self.max_cycles,
                             self.render_mode,
                             self.partial_observation,
                             self.trainable_cnn,
                             self._image_encoder,
                             self.image_encoder_batch_size,
                             self.image_encoder_use_cuda,
                             self._kwargs)

    def set_environment(self,
                        key,
                        max_cycles,
                        render_mode,
                        partial_observation,
                        trainable_cnn,
                        image_encoder,
                        image_encoder_batch_size,
                        image_encoder_use_cuda,
                        kwargs):

        # Convert list of kwargs to dictionary
        self.kwargs = kwargs
        self.get_kwargs(max_cycles, render_mode)

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
            # At each timestep we apply 4 pettingzoo timesteps,
            # in order to synchronize actions and obs
            self.episode_limit = int(self.episode_limit / 4)
        elif key == "space_invaders_v2":
            # At each timestep we apply 2 pettingzoo timesteps,
            # in order to synchronize actions and obs
            self.episode_limit = int(self.episode_limit / 2)

        self._env = ObservationPZ(self.__env,
                                  partial_observation,
                                  trainable_cnn,
                                  image_encoder,
                                  image_encoder_batch_size,
                                  image_encoder_use_cuda)
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
                                                        action.detach().cpu().item()
                   for action_idx, action in enumerate(actions)}

        self._obs, rewards, terminations, truncations, self._info = self._env.step(actions)

        # Add all rewards together
        reward = sum(rewards.values())
        done = (any([termination for termination in terminations.values()])
                or any([truncation for truncation in truncations.values()]))

        return float(reward), done, {}

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

        if self.trainable_cnn is False:
            return np.concatenate(self._obs, axis=0).astype(np.float32)
        else:
            return np.stack(self._obs, axis=0).astype(np.float32)

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


REGISTRY["pettingzoo"] = partial(env_fn, env=_PettingZooWrapper)
############################################

############################################
# Registration for Overcooked


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
        info["TimeLimit.truncated"] = False # There is no truncation in Overcooked
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
                 horizon,
                 seed,
                 reward_type):

        assert key in OVERCOOKED_KEY_CHOICES, \
            f"Invalid 'key': {key}! \nChoose one of the following: \n{OVERCOOKED_KEY_CHOICES}"
        self.key = key
        assert isinstance(horizon, int), f"Invalid horizon type: {type(horizon)}, 'horizon': {horizon}, is not 'int'!"
        self.horizon = horizon
        self._seed = seed
        assert reward_type in OVERCOOKED_REWARD_TYPE_CHOICES, \
            f"Invalid 'reward_type': {reward_type}! \nChoose one of the following: \n{OVERCOOKED_REWARD_TYPE_CHOICES}"
        self.reward_type = reward_type

        # Gym make
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        mdp = OvercookedGridworld.from_layout_name(self.key)
        base_env = OvercookedEnv.from_mdp(mdp, horizon=self.horizon)
        self.original_env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

        # Use the wrappers for handling the time limit and the environment observations properly.
        self.episode_limit = self.horizon
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
        return self.action_space * [1] # 1 indicates availability of action

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


REGISTRY["overcooked"] = partial(env_fn, env=_OvercookedWrapper)
############################################


############################################
# Registration for LBF

register(
    id="Foraging-8x8-5p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-11x11-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 2,
        "sight": 11,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-4f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 4,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": False,
    },
)

register(
    id="Foraging-15x15-3p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 4,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8s-25x25-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (25, 25),
        "max_food": 5,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5s-25x25-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (25, 25),
        "max_food": 5,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-50x50-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (50, 50),
        "max_food": 5,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-30x30-7p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 7,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 5,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-30x30-7p-4f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 7,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 4,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-4s-30x30-8p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 8,
        "max_player_level": 3,
        "field_size": (30, 30),
        "max_food": 5,
        "sight": 4,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-15x15-5p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 3,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-11x11-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-4s-11x11-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 2,
        "sight": 4,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-9x9-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-9x9-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 2,
        "sight": 9,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-8x8-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-6x6-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (6, 6),
        "max_food": 2,
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-5f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 5,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": False,
    },
)

register(
    id="Foraging-6x6-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (6, 6),
        "max_food": 1,
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7x7-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 1,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7x7-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 2,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-7x7-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-4p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-4p-2f-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": False,
    },
)

register(
    id="Foraging-8x8-4p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5x5-3p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 2,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5x5-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 1,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-6p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 6,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-15x15-3p-5f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (15, 15),
        "max_food": 5,
        "sight": 15,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-10x10-4p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (10, 10),
        "max_food": 1,
        "sight": 10,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7x7-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (7, 7),
        "max_food": 3,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-9x9-4p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 2,
        "sight": 9,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-7s-20x20-5p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (20, 20),
        "max_food": 3,
        "sight": 7,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5s-20x20-5p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 5,
        "max_player_level": 3,
        "field_size": (20, 20),
        "max_food": 3,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-11x11-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-4s-11x11-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 4,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-13x13-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 13,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-11x11-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (11, 11),
        "max_food": 3,
        "sight": 11,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-9x9-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (9, 9),
        "max_food": 3,
        "sight": 9,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-5x5-4p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 3,
        "sight": 5,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-10x10-3p-3f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (10, 10),
        "max_food": 3,
        "sight": 10,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-2s-12x12-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (12, 12),
        "max_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-6s-12x12-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (12, 12),
        "max_food": 2,
        "sight": 6,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-12x12-2p-2f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (12, 12),
        "max_food": 2,
        "sight": 12,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

register(
    id="Foraging-8x8-3p-1f-coop-v2",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 1,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
    },
)

############################################


############################################
# Registration for PressurePlate

# wraps original environment and adds the extra var "elapsed_time"
# to keep track of when an episode starts
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

    def __init__(self,
                 key,
                 horizon,
                 seed,
                 ):

        # Check key validity
        assert key in PRESSUREPLATE_KEY_CHOICES, \
            f"Invalid 'key': {key}! \nChoose one of the following: \n{PRESSUREPLATE_KEY_CHOICES}"
        self.key = key

        # Check horizon validity
        assert isinstance(horizon, int), f"Invalid horizon type: {type(horizon)}, 'horizon': {horizon}, is not 'int'!"

        # Default horizon
        if not horizon:
            horizon = 500 

        self.horizon = horizon
        self._seed = seed

        # Placeholders
        self.original_env = None
        self.episode_limit = None
        self._env = None
        self._obs = None
        self._info = None
        self.observation_space = None
        self.action_space = None

        # Gym make
        # base env sourced by gym.make with all its args
        from pressureplate.environment import PressurePlate
        self.original_env = gym.make(f"{key}")

        # Use the wrappers for handling the time limit and the environment observations properly.
        self.n_agents = self.original_env.n_agents
        self.episode_limit = self.horizon
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
        # By setting the "seed" in "np.random.seed" in "src/main.py" we control the randomness of the environment.
        self._seed = seed

        # Needed for rendering
        import cv2
        self.cv2 = cv2

    def step(self, actions):
        """ Returns reward, terminated, info """

        # Apply actions for each agent
        actions = [int(a) for a in actions]    
        # Make the environment step
        self._obs, rewards, terminations, self._info = self._env.step(actions)

        # Add all rewards together
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

    def reset(self):
        """ Returns initial observations and states """
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


REGISTRY["pressureplate"] = partial(env_fn, env=_PressurePlateWrapper)
############################################
