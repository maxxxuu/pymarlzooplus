# PyMARLzoo++

PyMARLzoo++ is an extension of [EPyMARL](https://github.com/uoe-agents/epymarl), and includes
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
- [PyMARLzoo++](#pymarlzoo)
- [Table of Contents](#table-of-contents)
- [Installation & Run instructions](#installation--run-instructions)
  - [Base requirements installation](#base-requirements-installation)
  - [Torch installation](#torch-installation)
  - [Torch-scatter installation](#torch-scatter-installation)
  - [Installing LBF, RWARE, MPE, PettingZoo, Overcooked, and Pressure plate](#installing-lbf-rware-mpe-pettingzoo-overcooked-and-pressureplate)
- [Run a hyperparameter search](#run-a-hyperparameter-search)
- [Saving and loading learnt models](#saving-and-loading-learnt-models)
  - [Saving models](#saving-models)
  - [Loading models](#loading-models)
- [Citing PyMARLzoo++, PyMARL, and EPyMARL](#citing-pymarlzoo-epymarl-and-pymarl)
- [License](#license)

# Installation & Run instructions
Note: ```pip install pymarlzooplusplus``` installation command will be available when the paper is accepted to not break the double-blind review process.

## Base requirements installation
To install the minimum requirements (without environments installation) run the following commands:
```sh
git clone ...
cd pymarlzooplusplus/installation
conda create -n pymarlzooplusplus python=3.8.18 -y
conda activate pymarlzooplusplus
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

## Installing LBF, RWARE, MPE, PettingZoo, Overcooked, PressurePlate and Capture Target

### LBF
To install [Level Based Foraging](https://github.com/uoe-agents/lb-foraging), run:
```sh
pip install lbforaging==1.1.1
```

Example of using LBF (replace ```<algo>``` and ```<scenario>```):
```sh
python3 src/main.py --config=<algo> --env-config=gymma with env_args.time_limit=25 env_args.key=<scenario>
```

Available scenarios we run experiments:
- "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
- "lbforaging:Foraging-2s-11x11-3p-2f-coop-v2",
- "lbforaging:Foraging-2s-8x8-3p-2f-coop-v2",
- "lbforaging:Foraging-2s-9x9-3p-2f-coop-v2",
- "lbforaging:Foraging-7s-20x20-5p-3f-coop-v2",
- "lbforaging:Foraging-2s-12x12-2p-2f-coop-v2",
- "lbforaging:Foraging-8s-25x25-8p-5f-coop-v2",
- "lbforaging:Foraging-7s-30x30-7p-4f-coop-v2"

### RWARE
To install [Multi-Robot Warehouse](https://github.com/uoe-agents/robotic-warehouse), run:
```sh
pip install rware==1.0.3
```

Example of using RWARE (replace ```<algo>``` and ```<scenario>```):
```sh
python3 src/main.py --config=<algo> --env-config=gymma with env_args.time_limit=500 env_args.key=<scenario>
```

Available scenarios we run experiments:
- "rware:rware-small-4ag-hard-v1",
- "rware:rware-tiny-4ag-hard-v1",
- "rware:rware-tiny-2ag-hard-v1"

### MPE
To install [Multi-agent Particle Environment](https://github.com/semitable/multiagent-particle-envs), being in ```pymarlzooplusplus/``` directory, run:
```sh
cd src/envs/multiagent-particle-envs/
pip install -e .
pip install seaborn==0.13.2  # Required for colors of landmarks
```

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

### PettingZoo
To install [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) run:
```sh
pip install opencv-python-headless==4.9.0.80
# or
pip install opencv-python==4.9.0.80 # only for rendering

pip install transformers==4.38.2 pettingzoo==1.24.3 'pettingzoo[atari]'==1.24.3 autorom==0.6.1 'pettingzoo[butterfly]'==1.24.3 
AutoROM -y
```
Example of using PettingZoo:
```sh
python3 src/main.py --config=qmix --env-config=pettingzoo with env_args.time_limit=900 env_args.key="pistonball_v6" env_args.kwargs="('n_pistons',10),"
```

### Overcooked
To install [Overcooked](https://github.com/HumanCompatibleAI/overcooked_ai), being in ```pymarlzooplusplus/``` directory, run:
```sh
cd src/envs/overcooked_ai/
pip install -e .
# Uninstall opencv because it installs newer version
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python-headless==4.9.0.80 # or pip install opencv-python==4.9.0.80 # only for rendering
```

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

### Pressure plate
To install [Pressure Plate](https://github.com/uoe-agents/pressureplate/), being in ```pymarlzooplusplus/``` directory, run:
```sh
cd src/envs/pressureplate_ai/
pip install -e .
pip install pyglet==1.5.27 # For rendering
```

Example of using Pressure plate (replace ```<algo>``` and ```<scenario>```):
```sh
python3 src/main.py --config=<algo> --env-config=pressureplate with env_args.key=<scenario> env_args.time_limit=500
```

Available scenarios we run experiments:
- "pressureplate-linear-4p-v0"
- "pressureplate-linear-6p-v0"

More available scenarios:
- "pressureplate-linear-5p-v0"

### Capture Target
To install [Capture Target](https://github.com/yuchen-x/MacDeepMARL/blob/master/src/rlmamr/my_env/capture_target.py), being in ```pymarlzooplusplus/``` directory run:
```sh
cd src/envs/capture_target/
pip install -e .
pip install pyglet==1.5.27 # For rendering
```

Example of using Capture Target (replace ```<algo>```):
```sh
python3 src/main.py --config=<algo> --env-config=capturetarget with env_args.key=<scenario> env_args.time_limit=60
```

Available scenarios:
- "CaptureTarget-6x6-1t-2a-v0"

### Box Pushing
To install [Box Pushing](https://github.com/yuchen-x/MacDeepMARL/blob/master/src/rlmamr/my_env/box_pushing.py), being in ```pymarlzooplusplus/``` directory run:
```sh
cd src/envs/box_pushing/
pip install -e .
pip install pyglet==1.5.27 # For rendering
```

Example of using Box Pushing (replace ```<algo>```):
```sh
python3 src/main.py --config=<algo> --env-config=boxpushing with env_args.key=<scenario> env_args.time_limit=60
```

Available scenarios:
- "BoxPushing-6x6-2a-v0"

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

# Citing PyMARLzoo++, EPyMARL and PyMARL

If you use PyMARLzoo++ in your research, please cite the ...

If you use the EPyMARL in your research, please cite the [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869).

*Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, & Stefano V. Albrecht. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021*

In BibTeX format:

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
All the source code that has been taken from the and EPyMARL PyMARL repository was licensed (and remains so) under the Apache License v2.0 (included in `LICENSE` file).
Any new code is also licensed under the Apache License v2.0
