# This is needed because it imports "envs" which is higher in directories hierarchy
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# Import packages
from envs import REGISTRY as env_REGISTRY
import random as rnd

# Arguments for PettingZoo
args = {
  "env": "pettingzoo",
  "env_args": {
      "key": "pistonball_v6",
      "time_limit": 900,
      "render_mode": "rgb_array",
      "image_encoder": "ResNet18",
      "image_encoder_use_cuda": True,
      "image_encoder_batch_size": 10,
      "centralized_image_encoding": False,
      "partial_observation": False,
      "trainable_cnn": False,
      "kwargs": "('n_pistons',10),",
      "seed": 2024
  }
}

# Arguments for Overcooked
# args = {
#   "env": "overcooked",
#   "env_args": {
#       "key": "cramped_room",
#       "time_limit": 500,
#       "reward_type": "sparse",
#       "seed": 2024
#   }
# }

# Arguments for Pressure Plate
# args = {
#   "env": "pressureplate",
#   "env_args": {
#       "key": "pressureplate-linear-4p-v0",
#       "time_limit": 500,
#       "seed": 2024
#   }
# }

# Arguments for LBF
# args = {
#   "env": "gymma",
#   "env_args": {
#       "key": "lbforaging:Foraging-8x8-2p-3f-v2",
#       "time_limit": 50,
#       "seed": 2024
#   }
# }

# Arguments for RWARE
# args = {
#   "env": "gymma",
#   "env_args": {
#       "key": "rware:rware-small-4ag-hard-v1",
#       "time_limit": 500,
#       "seed": 2024
#   }
# }

# Arguments for MPE
# args = {
#   "env": "gymma",
#   "env_args": {
#       "key": "mpe:SimpleSpread-3-v0",
#       "time_limit": 25,
#       "seed": 2024
#   }
# }

# Arguments for Capture Target
# args = {
#   "env": "capturetarget",
#   "env_args": {
#       "key": "CaptureTarget-6x6-1t-2a-v0",
#       "time_limit": 60,
#       "seed": 2024
#   }
# }

# Initialize environment
env = env_REGISTRY[args["env"]](**args["env_args"])
n_agns = env.n_agents
n_acts = env.get_total_actions()
# Reset the environment
obs, state = env.reset()
done = False
# Run an episode
t = 0
while not done:
    print(t)
    t += 1
    # Render the environment
    env.render()
    # Insert the policy's actions here
    actions = rnd.choices(range(0, n_acts), k=n_agns)
    # Apply an environment step
    reward, done, info = env.step(actions)
    obs = env.get_obs()
    state = env.get_state()
# Terminate the environment
env.close()
