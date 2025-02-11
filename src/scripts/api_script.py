# This is necessary because it imports "envs" which is higher in directories' hierarchy
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# Import packages
from envs import REGISTRY as env_REGISTRY

##################################### API for fully cooperative tasks. ###################################
##########################################################################################################
# ## 'reward' is the sum of all agents' rewards,
# ## 'done' is False if at least an agent's done is False,
# ## 'info' contains only 'TimeLimit.truncated'
# ## 'obs' is a list of numpy arrays each of which corresponds to an agent.
# ## 'state' is a single numpy array, i.e., the concatenation of all observations.
#
# # Arguments for PettingZoo.
# # Specifically: Butterfly (except from "Knights Archers Zombies"),
# #               Atari (only "Emtombed: Cooperative" and "Space Invaders")
# args = {
#   "env": "pettingzoo",
#   "env_args": {
#       "key": "pistonball_v6",
#       "time_limit": 900,  # Episode horizon.
#       "render_mode": "rgb_array",  # Options: "human", "rgb_array
#       "image_encoder": "ResNet18",  # Options: "ResNet18", "SlimSAM", "CLIP"
#       "image_encoder_use_cuda": True,  # Whether to load image-encoder in GPU or not.
#       "image_encoder_batch_size": 10,  # How many images to encode in a single pass.
#       "partial_observation": False,  # Only for "Emtombed: Cooperative" and "Space Invaders"
#       "trainable_cnn": False,  # Specifies whether to return image-observation or the encoded vector-observation
#       "kwargs": "('n_pistons',10),",
#       "seed": 2024
#   }
# }

# # Arguments for Overcooked
# args = {
#   "env": "overcooked",
#   "env_args": {
#       "key": "coordination_ring",
#       "time_limit": 500,
#       "reward_type": "sparse",
#       "seed": 2024
#   }
# }

# # Arguments for Pressure Plate
# args = {
#   "env": "pressureplate",
#   "env_args": {
#       "key": "pressureplate-linear-4p-v0",
#       "time_limit": 500,
#       "seed": 2024
#   }
# }

# # Arguments for LBF
# args = {
#   "env": "gymma",
#   "env_args": {
#       "key": "lbforaging:Foraging-8x8-2p-3f-v2",
#       "time_limit": 50,
#       "seed": 2024
#   }
# }

# # Arguments for RWARE
# args = {
#   "env": "gymma",
#   "env_args": {
#       "key": "rware:rware-small-4ag-hard-v1",
#       "time_limit": 500,
#       "seed": 2024
#   }
# }

# # Arguments for MPE
# args = {
#   "env": "gymma",
#   "env_args": {
#       "key": "mpe:SimpleSpeakerListener-v0",
#       "time_limit": 25,
#       "seed": 2024
#   }
# }

# # Arguments for Capture Target
# args = {
#   "env": "capturetarget",
#   "env_args": {
#       "key": "CaptureTarget-6x6-1t-2a-v0",
#       "time_limit": 60,
#       "seed": 2024
#   }
# }

# # Arguments for Box Pushing
# args = {
#   "env": "boxpushing",
#   "env_args": {
#       "key": "BoxPushing-6x6-2a-v0",
#       "time_limit": 60,
#       "seed": 2024
#   }
# }

# # Initialize environment
# env = env_REGISTRY[args["env"]](**args["env_args"])
# n_agns = env.n_agents
# n_acts = env.get_total_actions()
# # Reset the environment
# obs, state = env.reset()
# done = False
# # Run an episode
# while not done:
#     # Render the environment (optional)
#     env.render()
#     # Insert the policy's actions here
#     actions = env.sample_actions()
#     # Apply an environment step
#     reward, done, info = env.step(actions)  # In case
#     obs = env.get_obs()
#     state = env.get_state()
# # Terminate the environment
# env.close()

##########################################################################################################


####################################### API for NOT fully cooperative tasks, two cases: ###################################
# ## (a) For common observation space:
# ##    - 'reward' is returned as provided by PettingZoo, as well as 'done'.
# ##    - 'info' contains 'TimeLimit.truncated', 'truncations', and 'info' dictionaries as returned by PettingZoo
# ##    - 'obs' is returned as given by PettingZoo.
# ##    - 'state' is a single numpy array, i.e., the concatenation of all observations.
# ## (b) For NOT common observation space, the only difference is that:
# ##    - 'state' contains all the observations in a dictionary as given by PettingZoo.
#
# # Arguments for PettingZoo.
# # Specifically: Atari (except from "Emtombed: Cooperative" and "Space Invaders"),
# #               Butterfly (only "Knights Archers Zombies"),
# #               MPE, and SISL
# args = {
#   "env": "pettingzoo",
#   "env_args": {
#       "key": "basketball_pong_v3",
#       "time_limit": 900,
#       "render_mode": "rgb_array",
#       "image_encoder": "ResNet18",
#       "image_encoder_use_cuda": False,
#       "image_encoder_batch_size": 10,
#       "trainable_cnn": False,
#       "kwargs": "",
#       "seed": 2024
#   }
# }
#
# # Initialize environment
# env = env_REGISTRY[args["env"]](**args["env_args"])
# # Reset the environment
# obs, state = env.reset()
# done = False
# # Run an episode
# while not done:
#     # Render the environment (optional)
#     env.render()
#     # Insert the policy's actions here
#     actions = env.sample_actions()
#     # Apply an environment step
#     reward, done, info = env.step(actions)
#     done = all([agent_done for agent_done in done.values()])
#     obs = env.get_obs()
#     state = env.get_state()
# # Terminate the environment
# env.close()

##########################################################################################################


####################################### API for Classic environment ###################################
# ## We return the original PettingZoo environment.

# # Arguments for PettingZoo Classic
# args = {
#   "env": "pettingzoo",
#   "env_args": {
#       "key": "chess_v6",
#       "render_mode": "human",
#       "kwargs": "",
#   }
# }
#
# # Initialize environment
# env = env_REGISTRY[args["env"]](**args["env_args"]).original_env
# # Reset environment
# env.reset(seed=42)
# # Run an episode
# for agent in env.agent_iter():
#     # Render the environment (optional)
#     env.render()
#     # Get environment data
#     observation, reward, termination, truncation, info = env.last()
#     # Get action
#     if termination or truncation:
#         action = None
#     else:
#         mask = observation["action_mask"]  # For 'Rock Paper Scissors', comment out this line
#         # this is where you would insert your policy
#         action = env.action_space(agent).sample(mask) # For 'Rock Paper Scissors', don't use the 'mask'
#     # Apply an environment step
#     env.step(action)
# env.close()

##########################################################################################################