# Extended Python MARL framework - EPyMARL

EPyMARL is  an extension of [PyMARL](https://github.com/oxwhirl/pymarl), and includes
- **New!** Support for training in environments with individual rewards for all agents (for all algorithms that support such settings)
- Additional algorithms (IA2C, IPPO, MADDPG, MAA2C and MAPPO)
- Support for [Gym](https://github.com/openai/gym) environments (on top of the existing SMAC support)
- Option for no-parameter sharing between agents (original PyMARL only allowed for parameter sharing)
- Flexibility with extra implementation details (e.g. hard/soft updates, reward standarization, and more)
- Consistency of implementations between different algorithms (fair comparisons)

See our blog post here: https://agents.inf.ed.ac.uk/blog/epymarl/

## Update as of *June 2024*!

### Support for training in environments with individual rewards for all agents
Previously PyMARL and EPyMARL only supported training of MARL algorithms in common-reward environments. To support environments which naturally provide individual rewards for agents (e.g. LBF and RWARE), we previously scalarised the rewards of all agents using a sum operation to obtain a single common reward that was then given to all agents. We are glad to announce that EPyMARL now supports training in general-sum reward environments (for all algorithms that are sound to train in general-sum reward settings)!

- **Algorithms that support general-sum reward envs**: IA2C, IPPO, MAA2C, MAPPO, IQL, PAC
- Algorithms that only support common-reward envs: COMA, VDN, QMIX, QTRAN

By default, EPyMARL runs experiments with common rewards (as done previously). To run an experiment with individual rewards for all agents, set `common_reward=False`. For example to run MAPPO in a LBF task with individual rewards:
```sh
python3 src/main.py --config=mappo --env-config=gymma with env_args.time_limit=25 env_args.key="lbforaging:Foraging-8x8-2p-3f-v2" common_reward=False
```
When using the `common_reward=True` setup in environments which naturally provide individual rewards, by default we scalarise the rewards into a common reward by summing up all rewards. This is now configurable and we support the mean operation as an alternative scalarisation. To use the mean scalarisation, set `reward_scalarisation="mean"`.

### Plotting script
We have added a simple plotting script under `plot_results.py` to load data from sacred logs and visualise them for executed experiments. The script supports plotting of any logged metric, can apply simple window-smoothing, aggregates results across multiple runs of the same algorithm, and can filter which results to plot based on algorithm and environment names.

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
python3 main.py --config=pac_ns --env-config=gymma with env_args.time_limit=1 env_args.key=matrixgames:penalty-100-nostate-v0
```

# Table of Contents
- [Extended Python MARL framework - EPyMARL](#extended-python-marl-framework---epymarl)
- [Table of Contents](#table-of-contents)
- [Installation & Run instructions](#installation--run-instructions)
  - [Installing LBF, RWARE, and MPE](#installing-lbf-rware-and-mpe)
  - [Installing MARBLER for Sim2Real Evaluation](#installing-marbler)
  - [Using A Custom Gym Environment](#using-a-custom-gym-environment)
- [Run an experiment on a Gym environment](#run-an-experiment-on-a-gym-environment)
- [Run a hyperparameter search](#run-a-hyperparameter-search)
- [Saving and loading learnt models](#saving-and-loading-learnt-models)
  - [Saving models](#saving-models)
  - [Loading models](#loading-models)
- [Citing PyMARL and EPyMARL](#citing-pymarl-and-epymarl)
- [License](#license)

# Installation & Run instructions

For information on installing and using this codebase with SMAC, we suggest visiting and reading the original [PyMARL](https://github.com/oxwhirl/pymarl) README. Here, we maintain information on using the extra features EPyMARL offers. To install the codebase, clone this repo and run:
```sh
pip install -r requirements.txt
```

Note that the PAC algorithm and environments introduce additional dependencies. To install these dependencies, use the provided requirements files:
```sh
# install PAC dependencies
pip install -r pac_requirements.txt
# install environments
pip install -r env_requirements.txt
```

## Installing Environments

In [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869) we introduce the Level-Based Foraging (LBF) and Multi-Robot Warehouse (RWARE) environments, and additionally evaluate in SMAC, Multi-agent Particle environments. and a set of matrix games.

To install all environments, you can use the provided `env_requirements.txt`:
```sh
pip install -r env_requirements.txt
```
which will install LBF, RWARE, SMAC, our MPE form, and matrix games.


To install these individually, please visit:
- [Level Based Foraging](https://github.com/uoe-agents/lb-foraging) or install with `pip install lbforaging`
- [Multi-Robot Warehouse](https://github.com/uoe-agents/robotic-warehouse) or install with `pip install rware`
- [Our fork of MPE](https://github.com/semitable/multiagent-particle-envs), clone it and install it with `pip install -e .`
- [Matrix games](https://github.com/uoe-agents/matrix-games), clone it and install with `pip install -e .`

Example of using LBF:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="lbforaging:Foraging-8x8-2p-3f-v2"
```
Example of using RWARE:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v1"
```

For MPE, our fork is needed. Essentially all it does (other than fixing some gym compatibility issues) is i) registering the environments with the gym interface when imported as a package and ii) correctly seeding the environments iii) makes the action space compatible with Gym (I think MPE originally does a weird one-hot encoding of the actions).

The environments names in MPE are:
```
...
    "multi_speaker_listener": "MultiSpeakerListener-v0",
    "simple_adversary": "SimpleAdversary-v0",
    "simple_crypto": "SimpleCrypto-v0",
    "simple_push": "SimplePush-v0",
    "simple_reference": "SimpleReference-v0",
    "simple_speaker_listener": "SimpleSpeakerListener-v0",
    "simple_spread": "SimpleSpread-v0",
    "simple_tag": "SimpleTag-v0",
    "simple_world_comm": "SimpleWorldComm-v0",
...
```
Therefore, after installing them you can run it using:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:SimpleSpeakerListener-v0"
```

The pretrained agents are included in this repo [here](https://github.com/uoe-agents/epymarl/tree/main/src/pretrained). You can use them with:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:SimpleAdversary-v0" env_args.pretrained_wrapper="PretrainedAdversary"
```
and
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:SimpleTag-v0" env_args.pretrained_wrapper="PretrainedTag"
```

## Installing MARBLER

[MARBLER](https://github.com/GT-STAR-Lab/MARBLER) is a gym built for [the Robotarium](https://www.robotarium.gatech.edu) to enable free and effortless Sim2Real evaluation of algorithms. Clone it and follow the instructions on its Github to install it.

Example of using MARBLER:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=10000 env_args.key="robotarium_gym:PredatorCapturePrey-v0"
```

## Using A Custom Gym Environment

EPyMARL supports environments that have been registered with Gym. 
The only difference with the Gym framework would be that the returned rewards should be a tuple (one reward for each agent). In this cooperative framework we sum these rewards together.

Environments that are supported out of the box are the ones that are registered in Gym automatically. Examples are: [Level-Based Foraging](https://github.com/semitable/lb-foraging) and [RWARE](https://github.com/semitable/robotic-warehouse). 

To register a custom environment with Gym, use the template below (taken from Level-Based Foraging).
```python
from gym.envs.registration import registry, register, make, spec
register(
  id="Foraging-8x8-2p-3f-v2",                     # Environment ID.
  entry_point="lbforaging.foraging:ForagingEnv",  # The entry point for the environment class
  kwargs={
            ...                                   # Arguments that go to ForagingEnv's __init__ function.
        },
    )
```

# Run an experiment on a Gym environment

```shell
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v2"
```
 In the above command `--env-config=gymma` (in constrast to `sc2` will use a Gym compatible wrapper). `env_args.time_limit=50` sets the maximum episode length to 50 and `env_args.key="..."` provides the Gym's environment ID. In the ID, the `lbforaging:` part is the module name (i.e. `import lbforaging` will run automatically).


The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

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

Learnt models can be loaded using the `checkpoint_path` and `load_step` parameters. `checkpoint_path` should point to a directory stored for a run by epymarl as stated above. The pointed-to directory should contain sub-directories for various timesteps at which checkpoints were stored. If `load_step` is not provided (by default `load_step=0`) then the last checkpoint of the pointed-to run is loaded. Otherwise the checkpoint of the closest timestep to `load_step` will be loaded. After loading, the learning will proceed from the corresponding timestep.

To only evaluate loaded models without any training, set the `checkpoint_path` and `load_step` parameters accordingly for the loading, and additionally set `evaluate=True`. Then, the loaded checkpoint will be evaluated for `test_nepisode` episodes before terminating the run.

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
