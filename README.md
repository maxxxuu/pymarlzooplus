# PyMARLzoo+

PyMARLzoo+ is an extension of [EPyMARL](https://github.com/uoe-agents/epymarl), and includes
- Additional (7) algorithms: 
  - HAPPO, 
  - MAT-DEC, 
  - QPLEX, 
  - EOI, 
  - EMC, 
  - MASER, 
  - CDS
- Support for [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) environments
- Support for [Overcooked](https://github.com/HumanCompatibleAI/overcooked_ai) environments.
- Support for [Pressure plate](https://github.com/uoe-agents/pressureplate) environments.
- Support for [Capture Target](https://github.com/yuchen-x/MacDeepMARL/blob/master/src/rlmamr/my_env/capture_target.py) environment.
- Support for [Box Pushing](https://github.com/yuchen-x/MacDeepMARL/blob/master/src/rlmamr/my_env/box_pushing.py) environment.

Algorithms (9) maintained from EPyMARL:
- COMA
- QMIX
- MAA2C
- MAPPO
- VDN
- MADDPG
- IQL
- IPPO
- IA2C

# Table of Contents
- [PyMARLzoo+](#pymarlzoo)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [Base requirements installation](#base-requirements-installation)
  - [Torch installation](#torch-installation)
  - [Torch-scatter installation](#torch-scatter-installation)
  - [Installing LBF, RWARE, MPE, PettingZoo, Overcooked, and Pressure plate](#installing-lbf-rware-mpe-pettingzoo-overcooked-and-pressureplate)
- [Run instructions](#run-instructions)
- [Environment API](#environment-api)
- [Run a hyperparameter search](#run-a-hyperparameter-search)
- [Saving and loading learnt models](#saving-and-loading-learnt-models)
  - [Saving models](#saving-models)
  - [Loading models](#loading-models)
- [Citing PyMARLzoo+, PyMARL, and EPyMARL](#citing-pymarlzoo-epymarl-and-pymarl)
- [License](#license)

# Installation
Run:
```sh
pip install pymarlzooplus
``` 
NOTE: We work on this!

Otherwise, follow the instructions below to install from source.

## Base requirements installation
To install the minimum requirements (without environment installation), run the following commands:
```sh
git clone https://github.com/AILabDsUnipi/pymarlzooplus.git
cd pymarlzooplus/installation
conda create -n pymarlzooplus python=3.8.18 -y
conda activate pymarlzooplus
python3 -m pip install pip==24.0
pip install wheel==0.38.4 setuptools==65.5.0 einops
pip install -r requirements.txt
``` 
openCV:
```sh
pip install opencv-python==4.9.0.80
```
or, openCV for headless machines:
```sh
pip install opencv-python==4.9.0.80
```

## Torch installation
Then, run the following to install torch:
```sh
pip install protobuf==3.20.*
```
If you need torch with CUDA support:
```sh
pip3 install torch==2.1.2 torchvision==0.16.2--index-url https://download.pytorch.org/whl/cu121
```
otherwise:
```sh
pip3 install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
```

## Torch-scatter installation
Finally, run the following for torch-scatter installation (which is needed for most of the Actor-Critic algorithms):
```sh
sh ./install_torch_scatter.sh
```

## Installing LBF, RWARE, MPE, PettingZoo, Overcooked, PressurePlate, Capture Target and Box Pushing

### LBF
To install [Level Based Foraging](https://github.com/uoe-agents/lb-foraging), run:
```sh
pip install lbforaging==1.1.1
```

### RWARE
To install [Multi-Robot Warehouse](https://github.com/uoe-agents/robotic-warehouse), run:
```sh
pip install rware==1.0.3
```

### MPE
To install [Multi-agent Particle Environment](https://github.com/semitable/multiagent-particle-envs), being in ```pymarlzooplus/``` directory, run:
```sh
cd src/envs/multiagent-particle-envs/
pip install -e .
pip install seaborn==0.13.2  # Required for colors of landmarks
```

### PettingZoo
To install [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) run:
```sh
pip install opencv-python-headless==4.9.0.80
# or
pip install opencv-python==4.9.0.80 # only for rendering

pip install transformers==4.38.2 pettingzoo==1.24.3 'pettingzoo[atari]'==1.24.3 autorom==0.6.1 'pettingzoo[butterfly]'==1.24.3 'pettingzoo[classic]'==1.24.3 'pettingzoo[sisl]'==1.24.3
AutoROM -y
```

### Overcooked
To install [Overcooked](https://github.com/HumanCompatibleAI/overcooked_ai), being in ```pymarlzooplus/``` directory, run:
```sh
cd src/envs/overcooked_ai/
pip install -e .
# Uninstall opencv because it installs newer version
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python-headless==4.9.0.80 # or pip install opencv-python==4.9.0.80 # only for rendering
```

### Pressure plate
To install [Pressure Plate](https://github.com/uoe-agents/pressureplate/), being in ```pymarlzooplusplus/``` directory, run:
```sh
cd src/envs/pressureplate_ai/
pip install -e .
pip install pyglet==1.5.27 # For rendering
```

### Capture Target
To install [Capture Target](https://github.com/yuchen-x/MacDeepMARL/blob/master/src/rlmamr/my_env/capture_target.py), being in ```pymarlzooplus/``` directory run:
```sh
cd src/envs/capture_target/
pip install -e .
pip install pyglet==1.5.27 # For rendering
```

### Box Pushing
To install [Box Pushing](https://github.com/yuchen-x/MacDeepMARL/blob/master/src/rlmamr/my_env/box_pushing.py), being in ```pymarlzooplus/``` directory run:
```sh
cd src/envs/box_pushing/
pip install -e .
pip install pyglet==1.5.27 # For rendering
```

# Run instructions
To run an experiment, you can use the following command:

```sh
python3 src/main.py --config=<algo> --env-config=<config> with env_args.time_limit=<time_limit> env_args.key=<scenario>
```

The available algorithms (replace ```<algo>```) are the following:
- "coma"
- "qmix"
- "maa2c"
- "mappo"
- "vdn"
- "maddpg"
- "iql"
- "ippo"
- "ia2c"
- "happo" 
- "mat-dec" 
- "qplex" 
- "eoi" 
- "emc" 
- "maser" 
- "cds"

**These algorithms are compatible with our training framework for all the environments.**
Before running an experiment, you need to edit the corresponding configuration file in [src/config/algorithms](src/config/algs) to set the hyperparameters of the algorithm you want to run.

## LBF

Example of using LBF (replace ```<algo>``` and ```<scenario>```):
```sh
python3 src/main.py --config=<algo> --env-config=gymma with env_args.time_limit=25 env_args.key=<scenario>
```

Available scenarios (replace ```<scenario>```) we run experiments:
- "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
- "lbforaging:Foraging-2s-11x11-3p-2f-coop-v2",
- "lbforaging:Foraging-2s-8x8-3p-2f-coop-v2",
- "lbforaging:Foraging-2s-9x9-3p-2f-coop-v2",
- "lbforaging:Foraging-7s-20x20-5p-3f-coop-v2",
- "lbforaging:Foraging-2s-12x12-2p-2f-coop-v2",
- "lbforaging:Foraging-8s-25x25-8p-5f-coop-v2",
- "lbforaging:Foraging-7s-30x30-7p-4f-coop-v2"

All the above scenarios are compatible both with our **training framework** and the **environment API**.

Before running an experiment, edit the configuration file in [src/config/envs/gymma.yaml](src/config/envs/gymma.yaml).

## RWARE
Example of using RWARE (replace ```<algo>``` and ```<scenario>```):
```sh
python3 src/main.py --config=<algo> --env-config=gymma with env_args.time_limit=500 env_args.key=<scenario>
```

Available scenarios we run experiments:
- "rware:rware-small-4ag-hard-v1",
- "rware:rware-tiny-4ag-hard-v1",
- "rware:rware-tiny-2ag-hard-v1"

All the above scenarios are compatible both with our **training framework** and the **environment API**.

Before running an experiment, edit the configuration file in [src/config/envs/gymma.yaml](src/config/envs/gymma.yaml).

## MPE

Example of using MPE (replace ```<algo>``` and ```<scenario>```):
```sh
python3 src/main.py --config=<algo> --env-config=gymma with env_args.time_limit=25 env_args.key=<scenario>
```

Available scenarios we run experiments:
- "mpe:SimpleSpeakerListener-v0", 
- "mpe:SimpleSpread-3-v0",
- "mpe:SimpleSpread-4-v0",
- "mpe:SimpleSpread-5-v0",
- "mpe:SimpleSpread-8-v0"

More available scenarios:
- "mpe:MultiSpeakerListener-v0"
- "mpe:SimpleAdversary-v0",
- "mpe:SimpleCrypto-v0",
- "mpe:SimplePush-v0",
- "mpe:SimpleReference-v0",
- "mpe:SimpleTag-v0"
- "mpe:SimpleWorldComm-v0",
- "mpe:ClimbingSpread-v0"

All the above scenarios are compatible both with our **training framework** and the **environment API**.

Before running an experiment, edit the configuration file in [src/config/envs/gymma.yaml](src/config/envs/gymma.yaml).

## PettingZoo

Example of using PettingZoo (replace ```<algo>```, ```<time-limit>```, and ```<task>```):
```sh
python3 src/main.py --config=<algo> --env-config=pettingzoo with env_args.time_limit=<time-limit> env_args.key=<task>
```

Available tasks we run experiments:
- Atari:
  - "entombed_cooperative_v3"
  - "space_invaders_v2"
- Butterfly:
  - "pistonball_v6"
  - "cooperative_pong_v5"

The above tasks are compatible both with our **training framework** and the **environment API**.
Below, we list more tasks which are compatible only with the **environment API**:
- Atari:
  - "basketball_pong_v3"
  - "boxing_v2"
  - "combat_plane_v2"
  - "combat_tank_v2"
  - "double_dunk_v3"
  - "entombed_competitive_v3"
  - "flag_capture_v2"
  - "foozpong_v3"
  - "ice_hockey_v2"
  - "joust_v3"
  - "mario_bros_v3"
  - "maze_craze_v3"
  - "othello_v3"
  - "pong_v3"
  - "quadrapong_v4"
  - "space_war_v2"
  - "surround_v2"
  - "tennis_v3"
  - "video_checkers_v4"
  - "volleyball_pong_v3"
  - "warlords_v3"
  - "wizard_of_wor_v3"
- Butterfly:
  - "knights_archers_zombies_v10"
- Classic:
  - "chess_v6"
  - "connect_four_v3"
  - "gin_rummy_v4"
  - "go_v5"
  - "hanabi_v5"
  - "leduc_holdem_v4"
  - "rps_v2"
  - "texas_holdem_no_limit_v6"
  - "texas_holdem_v4"
  - "tictactoe_v3"
- MPE
  - "simple_v3"
  - "simple_adversary_v3"
  - "simple_crypto_v3"
  - "simple_push_v3"
  - "simple_reference_v3"
  - "simple_speaker_listener_v4"
  - "simple_spread_v3"
  - "simple_tag_v3"
  - "simple_world_comm_v3"
- SISL
  - "multiwalker_v9"
  - "pursuit_v4"
  - "waterworld_v4"

Before running an experiment, edit the configuration file in [src/config/envs/pettingzoo.yaml](src/config/envs/pettingzoo.yaml). Also, there you can find useful comments for each task's arguments.

## Overcooked
Example of using Overcooked (replace ```<algo>```, ```<scenario>```, and ```<reward_type>```):
```sh
python3 src/main.py --config=<algo> --env-config=overcooked with env_args.time_limit=500 env_args.key=<scenario> env_args.reward_type=<reward_type>
```

Available scenarios we run experiments:
- "cramped_room"
- "asymmetric_advantages"
- "coordination_ring"

More available scenarios:
- "counter_circuit"
- "random3"
- "random0"
- "unident"
- "forced_coordination"
- "soup_coordination"
- "small_corridor"
- "simple_tomato"
- "simple_o_t"
- "simple_o"
- "schelling_s"
- "schelling"
- "m_shaped_s"
- "long_cook_time"
- "large_room"
- "forced_coordination_tomato"
- "cramped_room_tomato"
- "cramped_room_o_3orders"
- "asymmetric_advantages_tomato"
- "bottleneck"
- "cramped_corridor"
- "counter_circuit_o_1order"

Reward types available:
- "sparse" (we used it to run our experiments)
- "shaped"

All the above scenarios are compatible both with our **training framework** and the **environment API**.

Before running an experiment, edit the configuration file in [src/config/envs/overcooked.yaml](src/config/envs/overcooked.yaml). 

## Pressure Plate
Example of using Pressure plate (replace ```<algo>``` and ```<scenario>```):
```sh
python3 src/main.py --config=<algo> --env-config=pressureplate with env_args.key=<scenario> env_args.time_limit=500
```

Available scenarios we run experiments:
- "pressureplate-linear-4p-v0"
- "pressureplate-linear-6p-v0"

More available scenarios:
- "pressureplate-linear-5p-v0"

Before running an experiment, edit the configuration file in [src/config/envs/pressureplate.yaml](src/config/envs/pressureplate.yaml).

## Capture target
Example of using Capture Target (replace ```<algo>``` and ```<scenario>```):
```sh
python3 src/main.py --config=<algo> --env-config=capturetarget with env_args.key=<scenario> env_args.time_limit=60
```

Available scenario:
- "CaptureTarget-6x6-1t-2a-v0"

The above scenario is compatible both with our **training framework** and the **environment API**.

Before running an experiment, edit the configuration file in [src/config/envs/capturetarget.yaml](src/config/envs/capturetarget.yaml). Also, there you can find useful comments for the rest of arguments.

## Box Pushing
Example of using Box Pushing (replace ```<algo>``` and ```<scenario>```):
```sh
python3 src/main.py --config=<algo> --env-config=boxpushing with env_args.key=<scenario> env_args.time_limit=60
```

Available scenario:
- "BoxPushing-6x6-2a-v0"

The above scenario is compatible both with our **training framework** and the **environment API**.

Before running an experiment, edit the configuration file in [src/config/envs/boxpushing.yaml](src/config/envs/boxpushing.yaml).


# Environment API
Except from using our training framework, you can use PyMARLzoo+ in your custom training pipeline as follows:
```python
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

# Initialize environment
env = env_REGISTRY[args["env"]](**args["env_args"])
# Reset the environment
obs, state = env.reset()
done = False
# Run an episode
while not done:
    # Render the environment (optional)
    env.render()
    # Insert the policy's actions here
    actions = env.sample_actions()
    # Apply an environment step
    reward, done, info = env.step(actions)
    obs = env.get_obs()
    state = env.get_state()
# Terminate the environment
env.close()
```
The code above can be used as is (changing the arguments accordingly) **with the fully cooperative tasks** (see section [Run instructions](#run-instructions) to find out which tasks are fully cooperative).

In the file [src/scripts/api_script.py](src/scripts/api_script.py), you can find a complete example for all the supported environments.

For a description of each environment's arguments, please see the corresponding config file in [src/config/envs](src/config/envs). 

# Run a hyperparameter search

We include a script named `search.py` which reads a search configuration file (e.g. the included `search.config.example.yaml`) and runs a hyperparameter search in one or more tasks. The script can be run using
```shell
python search.py run --config=search.config.example.yaml --seeds 5 locally
```
In a cluster environment where one run should go to a single process, it can also be called in a batch script like:
```shell
python search.py run --config=search.config.example.yaml --seeds 5 single 1
```
where the 1 is an index to the particular hyperparameter configuration and can take values from 1 to the number of different combinations.

# Saving and loading learnt models

## Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

## Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

# Citing PyMARLzoo+, EPyMARL and PyMARL

If you use PyMARLzoo+ in your research, please cite the [An Extended Benchmarking of Multi-Agent Reinforcement Learning Algorithms in Complex Fully Cooperative Tasks
]([https://arxiv.org/abs/2006.07869](https://www.arxiv.org/pdf/2502.04773)).

*George Papadopoulos, Andreas Kontogiannis, Foteini Papadopoulou, Chaido Poulianou, Ioannis Koumentis, George Vouros. An Extended Benchmarking of Multi-Agent Reinforcement Learning Algorithms in Complex Fully Cooperative Tasks, AAMAS 2025.*

If you use PyMARLzoo+ in your research, please cite the following:

```tex
@article{papadopoulos2025extended,
  title={An Extended Benchmarking of Multi-Agent Reinforcement Learning Algorithms in Complex Fully Cooperative Tasks},
  author={Papadopoulos, George and Kontogiannis, Andreas and Papadopoulou, Foteini and Poulianou, Chaido and Koumentis, Ioannis and Vouros, George},
  journal={arXiv preprint arXiv:2502.04773},
  year={2025}
}
```

*Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, & Stefano V. Albrecht. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021*

If you use EPyMARL in your research, please cite the following:

```tex
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   url = {http://arxiv.org/abs/2006.07869},
   openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
   code = {https://github.com/uoe-agents/epymarl},
}
```

If you use the original PyMARL in your research, please cite the [SMAC paper](https://arxiv.org/abs/1902.04043).

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

# License
All the source code that has been taken from the EPyMARL and PyMARL repository was licensed (and remains so) under the Apache License v2.0 (included in `LICENSE` file).
Any new code is also licensed under the Apache License v2.0
