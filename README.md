# Extended Python MARL framework - EPyMARL

EPyMARL is  an extension of [PyMARL](https://github.com/oxwhirl/pymarl), and includes
- Additional algorithms (IA2C, IPPO, MADDPG, MAA2C and MAPPO)
- Support for [Gym](https://github.com/openai/gym) environments (on top of the existing SMAC support)
- Option for no-parameter sharing between agents (original PyMARL only allowed for parameter sharing)
- Flexibility with extra implementation details (e.g. hard/soft updates, reward standarization, and more)

## Installation & Run instructions

For information on installing and using this codebase with SMAC, we suggest visiting and reading the original [PyMARL](https://github.com/oxwhirl/pymarl) README. Here, we maintain information on using the extra features EPyMARL offers.
To install the codebase, clone this repo and install the `requirements.txt`.  

### Using A Gym Environment

EPyMARL supports environments that have been registered with Gym. 
The only difference with the Gym framework would be that the returned rewards should be a tuple (one reward for each agent). In this cooperative framework we sum these rewards together.

Environments that are supported out of the box are the ones that are registered in Gym automatically. Examples are: [Level-Based Foraging](https://github.com/semitable/lb-foraging) and [RWARE](https://github.com/semitable/robotic-warehouse). 

To register a custom environment with Gym, use the template below (taken from Level-Based Foraging).
```python
from gym.envs.registration import registry, register, make, spec
register(
  id="Foraging-8x8-2p-3f-v0",                     # Environment ID.
  entry_point="lbforaging.foraging:ForagingEnv",  # The entry point for the environment class
  kwargs={
            ...                                   # Arguments that go to ForagingEnv's __init__ function.
        },
    )
```

## Run an experiment on a Gym environment

```shell
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v0"
```
 In the above command `--env-config=gymma` (in constrast to `sc2` will use a Gym compatible wrapper). `env_args.time_limit=50` sets the maximum episode length to 50 and `env_args.key="..."` provides the Gym's environment ID. In the ID, the `lbforaging:` part is the module name (i.e. `import lbforaging` will run automatically).


The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Citing PyMARL and EPyMARL

The Extended PyMARL (EPyMARL) codebase was discussed in [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869).

*Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, & Stefano V. Albrecht. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks, CoRR abs/2006.07869, 2021*

In BibTeX format:

```tex
@misc{papoudakis2021benchmarking,
  title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks}, 
  author={Georgios Papoudakis and Filippos Christianos and Lukas Sch\"afer and Stefano V. Albrecht},
  year={2021},
  eprint={2006.07869},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
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

## License
All the source code that has been taken from the PyMARL repository was licensed (and remains so) under the Apache License v2.0 (included in `LICENSE` file).
Any new code is also licensed under the Apache License v2.0
