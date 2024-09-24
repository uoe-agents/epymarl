# Extended Python MARL framework - EPyMARL

EPyMARL is  an extension of [PyMARL](https://github.com/oxwhirl/pymarl), and includes
- **New!** Support for training in environments with individual rewards for all agents (for all algorithms that support such settings)
- **New!** Updated EPyMARL to use maintained [Gymnasium](https://gymnasium.farama.org/index.html) library instead of deprecated OpenAI Gym version 0.21.
- **New!** Support for new environments: native integration of [PettingZoo](https://pettingzoo.farama.org/), [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator), [matrix games](https://github.com/uoe-agents/matrix-games), [SMACv2](https://github.com/oxwhirl/smacv2), and [SMAClite](https://github.com/uoe-agents/smaclite)
- **New!** Support for logging to [weights and biases (W&B)](https://wandb.ai/)
- **New!** We added a simple plotting script to visualise run data
- Additional algorithms (IA2C, IPPO, MADDPG, MAA2C and MAPPO)
- Option for no-parameter sharing between agents (original PyMARL only allowed for parameter sharing)
- Flexibility with extra implementation details (e.g. hard/soft updates, reward standarization, and more)
- Consistency of implementations between different algorithms (fair comparisons)

See our blog post here: https://agents.inf.ed.ac.uk/blog/epymarl/

## Update as of *July 2024*!

### Update to Gymnasium
It became increasingly difficult to install and rely on the deprecated OpenAI Gym version 0.21 EPyMARL previously depended on, so we moved EPyMARL to use the maintained [Gymnasium](https://gymnasium.farama.org/index.html) library and API. This move required updating of several environments that were built to work with EPyMARL's `gymma` wrapper, including [level-based foraging](https://github.com/uoe-agents/lb-foraging) and [multi-robot warehouse](https://github.com/uoe-agents/robotic-warehouse). Alongside this update to EPyMARL, we therefore also updated these environments as well as [SMAClite](https://github.com/uoe-agents/smaclite), [matrix games](https://github.com/uoe-agents/matrix-games), and wrote wrappers to maintain compatibility with [SMAC](https://github.com/oxwhirl/smac) and added integration for [SMACv2](https://github.com/oxwhirl/smacv2). We hope these changes will simplify integration of new environments and ensure that EPyMARL remains usable for a longer time.

To use the legacy version of EPyMARL with OpenAI Gym version 0.21, please use the previous version `v1.0.0` of EPyMARL.

For more information on how to install and run experiments in these environments, see [the documentation here](#installation--run-instructions).


### Support for training in environments with individual rewards for all agents
Previously EPyMARL only supported training of MARL algorithms in common-reward environments. To support environments which naturally provide individual rewards for agents (e.g. LBF and RWARE), we previously scalarised the rewards of all agents using a sum operation to obtain a single common reward that was then given to all agents. We are glad to announce that EPyMARL now supports training in general-sum reward environments (for all algorithms that are sound to train in general-sum reward settings)!

- **Algorithms that support general-sum reward envs**: IA2C, IPPO, MAA2C, MAPPO, IQL, PAC
- Algorithms that only support common-reward envs: COMA, VDN, QMIX, QTRAN

By default, EPyMARL runs experiments with common rewards (as done previously). To run an experiment with individual rewards for all agents, set `common_reward=False`. For example to run MAPPO in a LBF task with individual rewards:
```sh
python src/main.py --config=mappo --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3" common_reward=False
```
When using the `common_reward=True` setup in environments which naturally provide individual rewards, by default we scalarise the rewards into a common reward by summing up all rewards. This is now configurable and we support the mean operation as an alternative scalarisation. To use the mean scalarisation, set `reward_scalarisation="mean"`.

### Weights and Biases (W&B) Logging
We now support logging to W&B! To log data to W&B, you need to install the library with `pip install wandb` and setup W&B (see their [documentation](https://docs.wandb.ai/quickstart)). After, follow [our instructions](#weights-and-biases).

### Plotting script
We have added a simple plotting script under `plot_results.py` to load data from sacred logs and visualise them for executed experiments. For more details, see [the documentation here](#plotting).


## Update as of *15th July 2023*!
We have released our _Pareto Actor-Critic_ algorithm, accepted in TMLR, as part of the E-PyMARL source code. 

Find the paper here: https://arxiv.org/abs/2209.14344

Pareto-AC (Pareto-AC), is an actor-critic algorithm that utilises a simple principle of no-conflict games (and, in turn, cooperative games with identical rewards): each agent can assume the others will choose actions that will lead to a Pareto-optimal equilibrium.
Pareto-AC works especially well in environments with multiple suboptimal equilibria (a problem is also known as relative over-generalisation). We have seen impressive results in a diverse set of multi-agent games with suboptimal equilibria, including the matrix games of the MARL benchmark, but also LBF variations with high penalties.

PAC introduces additional dependencies specified in `pac_requirements.txt`. To install its dependencies, run
```sh
pip install -r pac_requirements.txt
```

To run Pareto-AC in an environment, for example the Penalty game, you can run:
```sh
python src/main.py --config=pac_ns --env-config=gymma with env_args.time_limit=1 env_args.key=matrixgames:penalty-100-nostate-v0
```

# Table of Contents
- [Extended Python MARL framework - EPyMARL](#extended-python-marl-framework---epymarl)
- [Table of Contents](#table-of-contents)
- [Installation & Run instructions](#installation--run-instructions)
  - [Installing Dependencies](#installing-dependencies)
  - [Benchmark Paper Experiments](#benchmark-paper-experiments)
  - [Experiments in SMACv2 and SMAClite](#experiments-in-smacv2-and-smaclite)
  - [Experiments in PettingZoo and VMAS](#experiments-in-pettingzoo-and-vmas)
  - [Registering and Running Experiments in Custom Environments](#registering-and-running-experiments-in-custom-environments)
- [Experiment Configurations](#experiment-configurations)
- [Run a hyperparameter search](#run-a-hyperparameter-search)
- [Logging](#logging)
  - [Weights and Biases](#weights-and-biases)
- [Saving and loading learnt models](#saving-and-loading-learnt-models)
  - [Saving models](#saving-models)
  - [Loading models](#loading-models)
- [Plotting](#plotting)
- [Citing PyMARL and EPyMARL](#citing-pymarl-and-epymarl)
- [License](#license)

# Installation & Run instructions

## Installing Dependencies

To install the dependencies for the codebase, clone this repo and run:
```sh
pip install -r requirements.txt
```

To install a set of supported environments, you can use the provided `env_requirements.txt`:
```sh
pip install -r env_requirements.txt
```
which will install the following environments:
- [Level Based Foraging](https://github.com/uoe-agents/lb-foraging)
- [Multi-Robot Warehouse](https://github.com/uoe-agents/robotic-warehouse)
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) (used for the multi-agent particle environment)
- [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator)
- [Matrix games](https://github.com/uoe-agents/matrix-games)
- [SMAC](https://github.com/oxwhirl/smac)
- [SMACv2](https://github.com/oxwhirl/smacv2)
- [SMAClite](https://github.com/uoe-agents/smaclite)

To install these environments individually, please see instructions in the respective repositories. We note that in particular SMAC and SMACv2 require a StarCraft II installation with specific map files. See their documentation for more details.

Note that the [PAC algorithm](#update-as-of-15th-july-2023) introduces separate dependencies. To install these dependencies, use the provided requirements file:
```sh
pip install -r pac_requirements.txt
```

## Benchmark Paper Experiments

In ["Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks"](https://arxiv.org/abs/2006.07869) we introduce the Level-Based Foraging (LBF) and Multi-Robot Warehouse (RWARE) environments, and additionally evaluate in SMAC, Multi-agent Particle environments, and a set of matrix games. After installing these environments (see instructions above), we can run experiments in these environments as follows:

Matrix games:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="matrixgames:penalty-100-nostate-v0"
```

LBF:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3"
```

RWARE:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v2"
```

MPE:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3"
```
Note that for the MPE environments tag (predator-prey) and adversary, we provide pre-trained prey and adversary policies. These can be used to control the respective agents to make these tasks fully cooperative (used in the paper) by setting `env_args.pretrained_wrapper="PretrainedTag"` or `env_args.pretrained_wrapper="PretrainedAdversary"`.

SMAC:
```sh
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name="3s5z"
```

Below, we provide the base environment and key / map name for all the environments evaluated in the "Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks":

- Matrix games: all with `--env-config=gymma with env_args.time_limit=25 env_args.key="..."`
  - Climbing: `matrixgames:climbing-nostate-v0`
  - Penalty $k=0$: `matrixgames:penalty-0-nostate-v0`
  - Penalty $k=-25$: `matrixgames:penalty-25-nostate-v0`
  - Penalty $k=-50$: `matrixgames:penalty-50-nostate-v0`
  - Penalty $k=-75$: `matrixgames:penalty-75-nostate-v0`
  - Penalty $k=-100$: `matrixgames:penalty-100-nostate-v0`
- LBF: all with `--env-config=gymma with env_args.time_limit=50 env_args.key="..."`
  - 8x8-2p-2f-coop: `lbforaging:Foraging-8x8-2p-2f-coop-v3`
  - 8x8-2p-2f-2s-coop: `lbforaging:Foraging-2s-8x8-2p-2f-coop-v3`
  - 10x10-3p-3f: `lbforaging:Foraging-10x10-3p-3f-v3`
  - 10x10-3p-3f-2s: `lbforaging:Foraging-2s-10x10-3p-3f-v3`
  - 15x15-3p-5f: `lbforaging:Foraging-15x15-3p-5f-v3`
  - 15x15-4p-3f: `lbforaging:Foraging-15x15-4p-3f-v3`
  - 15x15-4p-5f: `lbforaging:Foraging-15x15-4p-5f-v3`
- RWARE: all with `--env-config=gymma with env_args.time_limit=500 env_args.key="..."`
  - tiny 2p: `rware:rware-tiny-2ag-v2`
  - tiny 4p: `rware:rware-tiny-4ag-v2`
  - small 4p: `rware:rware-small-4ag-v2`
- MPE: all with `--env-config=gymma with env_args.time_limit=25 env_args.key="..."`
  - simple speaker listener: `pz-mpe-simple-speaker-listener-v4`
  - simple spread: `pz-mpe-simple-spread-v3`
  - simple adversary: `pz-mpe-simple-adversary-v3` with additional `env_args.pretrained_wrapper="PretrainedAdversary"`
  - simple tag: `pz-mpe-simple-tag-v3` with additional `env_args.pretrained_wrapper="PretrainedTag"`
- SMAC: all with `--env-config=sc2 with env_args.map_name="..."`
  - 2s_vs_1sc: `2s_vs_1sc`
  - 3s5z: `3s5z`
  - corridor: `corridor`
  - MMM2: `MMM2`
  - 3s_vs_5z: `3s_vs_5z`
  
## Experiments in SMACv2 and SMAClite

EPyMARL now supports the new SMACv2 and SMAClite environments. We provide wrappers to integrate these environments into the Gymnasium interface of EPyMARL. To run experiments in these environments, you can use the following exemplary commands:

SMACv2:
```sh
python src/main.py --config=qmix --env-config=sc2v2 with env_args.map_name="protoss_5_vs_5"
```
We provide prepared configs for a range of SMACv2 scenarios, as described in the [SMACv2 repository](https://github.com/oxwhirl/smacv2), under `src/config/envs/smacv2_configs`. These can be run by providing the name of the config file as the `env_args.map_name` argument. To define a new scenario, you can create a new config file in the same format as the provided ones and provide its name as the `env_args.map_name` argument.

SMAClite:
```sh
python src/main.py --config=qmix --env-config=smaclite with env_args.time_limit=150 env_args.map_name="MMM"
```
By default, SMAClite uses a numpy implementation of the RVO2 library for collision avoidance. To instead use a faster optimised C++ RVO2 library, follow the instructions of [this repo](https://github.com/micadam/SMAClite-Python-RVO2) and provide the additional argument `env_args.use_cpp_rvo2=True`.

## Experiments in PettingZoo and VMAS

EPyMARL supports the PettingZoo and VMAS libraries for multi-agent environments using wrappers. To run experiments in these environments, you can use the following exemplary commands:

PettingZoo:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3"
```

VMAS:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=150 env_args.key="vmas-balance"
```

## Registering and Running Experiments in Custom Environments

EPyMARL supports environments that have been registered with Gymnasium. If you would like to use any other Gymnasium environment, you can do so by using the `gymma` environment with the `env_args.key` argument being provided with the registration ID of the environment. Environments can either provide a single scalar reward to run common reward experiments (`common_reward=True`), or should provide one environment per agent to run experiments with individual rewards (`common_reward=False`) or with common rewards using some reward scalarisation (see [documentation](#support-for-training-in-environments-with-individual-rewards-for-all-agents) for more details). 

To register a custom environment with Gymnasium, use the template below:
```python
from gymnasium import register

register(
  id="my-environment-v1",                         # Environment ID.
  entry_point="myenv.environment:MyEnvironment",  # The entry point for the environment class
  kwargs={
            ...                                   # Arguments that go to MyEnvironment's __init__ function.
        },
    )
```

After, you can run an experiment in this environment using the following command:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="myenv:my-environment-v1"
```
assuming that the environment is registered with the ID `my-environment-v1` in the installed library `myenv`.

# Experiment Configurations

EPyMARL defines yaml configuration files for algorithms and environments under `src/config`. `src/config/default.yaml` defines default values for a range of configuration options, including experiment information (`t_max` for number of timesteps of training etc.) and algorithm hyperparameters.

Further environment configs (provided to the main script via `--env-config=...`) can be found in `src/config/envs`. Algorithm configs specifying algorithms and their hyperparameters (provided to the main script via `--config=...`) can be found in `src/config/algs`. To change hyperparameters or define a new algorithm, you can modify these yaml config files or create new ones.

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

# Logging

By default, EPyMARL will use sacred to log results and models to the `results` directory. These logs include configuration files, a json of all metrics, a txt file of all outputs and more. Additionally, EPyMARL can log data to tensorboard files by setting `use_tensorboard: True` in the yaml config. We also added support to log data to [weights and biases (W&B)](https://wandb.ai/) with instructions below.

## Weights and Biases

First, make sure to install W&B and follow their instructions to authenticate and setup your W&B library (see the [quickstart guide](https://docs.wandb.ai/quickstart) for more details).

To tell EPyMARL to log data to W&B, you then need to specify the following parameters in [your configuration](#experiment-configurations):
```yaml
use_wandb: True # Log results to W&B
wandb_team: null # W&B team name
wandb_project: null # W&B project name
```
to specify the team and project you wish to log to within your account, and set `use_wandb=True`. By default, we log all W&B runs in "offline" mode, i.e. the data will only be stored locally and can be uploaded to your W&B account via `wandb sync ...`. To directly log runs online, please specify `wandb_mode="online"` within the config.

We also support logging all stored models directly to W&B so you can download and inspect these from the W&B online dashboard. To do so, use the following config parameters:
```yaml
wandb_save_model: True # Save models to W&B (only done if use_wandb is True and save_model is True)
save_model: True # Save the models to disk
save_model_interval: 50000
```
Note that models are only saved in general if `save_model=True` and to further log them to W&B you need to specify `use_wandb`, `wandb_team`, `wandb_project`, and `wandb_save_model=True`.

# Saving and loading learnt models

## Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

## Loading models

Learnt models can be loaded using the `checkpoint_path` and `load_step` parameters. `checkpoint_path` should point to a directory stored for a run by epymarl as stated above. The pointed-to directory should contain sub-directories for various timesteps at which checkpoints were stored. If `load_step` is not provided (by default `load_step=0`) then the last checkpoint of the pointed-to run is loaded. Otherwise the checkpoint of the closest timestep to `load_step` will be loaded. After loading, the learning will proceed from the corresponding timestep.

To only evaluate loaded models without any training, set the `checkpoint_path` and `load_step` parameters accordingly for the loading, and additionally set `evaluate=True`. Then, the loaded checkpoint will be evaluated for `test_nepisode` episodes before terminating the run.

# Plotting

The plotting script provided as `plot_results.py` supports plotting of any logged metric, can apply simple window-smoothing, aggregates results across multiple runs of the same algorithm, and can filter which results to plot based on algorithm and environment names.

If multiple configs of the same algorithm exist within the loaded data and you only want to plot the best config per algorithm, then add the `--best_per_alg` argument! If this argument is not set, the script will visualise all configs of each (filtered) algorithm and show the values of the hyperparameter config that differ across all present configs in the legend.

# Citing EPyMARL and PyMARL

The Extended PyMARL (EPyMARL) codebase was used in [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869).

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
All the source code that has been taken from the PyMARL repository was licensed (and remains so) under the Apache License v2.0 (included in `LICENSE` file).
Any new code is also licensed under the Apache License v2.0
