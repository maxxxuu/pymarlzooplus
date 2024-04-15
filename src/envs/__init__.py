from functools import partial

from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
import gym
from gym import ObservationWrapper, spaces
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit

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
    def __init__(self, env, max_episode_steps):
        super().__init__(env)

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None # This is initialized to 0 by GymTimeLimit when reset() is calling.

    def step(self, actions):
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"

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

    def __init__(self, env, trainable_cnn, image_encoder, image_encoder_batch_size, image_encoder_use_cuda):
        super(ObservationPZ, self).__init__(env)

        self.trainable_cnn = trainable_cnn
        self.original_observation_space = self.env.observation_space(self.env.possible_agents[0])
        self.original_observation_space_shape = self.original_observation_space.shape
        self.is_image = len(self.original_observation_space_shape) == 3 and self.original_observation_space_shape[2] == 3
        assert self.is_image, f"Only images are supported, shape: {self.original_observation_space_shape}"

        ## Define image encoder. In this case, a pretrained model is used frozen, i.e., without further training.
        self.image_encoder = None
        if self.is_image and self.trainable_cnn is False:

            import torch
            self.torch = torch

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
                print("\nLoading pretrained ResNet18 model...")
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
                print("\nLoading pretrained SlimSAM model...")
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
                print("\nLoading pretrained CLIP-image-encoder model...")
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
            import torch
            self.torch = torch
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            img_size = 40 #TODO change it to 224
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=img_size, interpolation=1), # Resize the longest side to 224
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=(0, 0, 0)), # Pad to make the image square
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


class _PettingZooWrapper(MultiAgentEnv):
    def __init__(self,
                 key,
                 max_cycles,
                 seed,
                 render_mode,
                 trainable_cnn,
                 image_encoder,
                 image_encoder_batch_size,
                 image_encoder_use_cuda,
                 kwargs):

        self.key = key
        self.max_cycles = max_cycles
        self._seed = seed
        self.render_mode = render_mode
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
                             self.trainable_cnn,
                             self._image_encoder,
                             self.image_encoder_batch_size,
                             self.image_encoder_use_cuda,
                             self._kwargs)

    def set_environment(self,
                        key,
                        max_cycles,
                        render_mode,
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
        self.n_agents = self.original_env.max_num_agents
        setattr(self.original_env, 'spec', None)  # Just for support with "ObservationWrapper"
        self._env = TimeLimitPZ(self.original_env, max_episode_steps=self.episode_limit)
        self._env = ObservationPZ(self._env, trainable_cnn, image_encoder, image_encoder_batch_size, image_encoder_use_cuda)

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

    def step(self, actions):
        """ Returns reward, terminated, info """
        # Apply action for each agent
        actions = {f"{self.action_prefix[action_idx]}": action.item() if isinstance(action, np.int64)
                                                                      else
                                                        action.detach().cpu().item()
                   for action_idx, action in enumerate(actions)}

        self._obs, rewards, terminations, truncations, self._info = self._env.step(actions)

        # Add all rewards together
        reward = sum(rewards.values())
        done = any([termination for termination in terminations.values()]) or any([truncation for truncation in truncations.values()])

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
        self._obs = self._env.reset()

        return self.get_obs(), self.get_state()

    def render(self):
        if self.render_mode != "human": # otherwise it is already rendered
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
        assert self.agent_policy_idx == 1-self.other_agent_idx
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

        assert key in OVERCOOKED_KEY_CHOICES, f"Invalid 'key': {key}! \nChoose one of the following: \n{OVERCOOKED_KEY_CHOICES}"
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
        self.n_agents = 2 # Always 2 agents
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
            actions = actions[::-1] # reverse the order

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




