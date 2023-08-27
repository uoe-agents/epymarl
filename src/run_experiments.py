#from iql import run_training
from iql_gym import run_training
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from pathlib import Path
import os 
import datetime
from distutils.util import strtobool
import argparse 
import yaml

sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize':(11.7,8.27)})
#import warnings
#warnings.filterwarnings("ignore")

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-buffer", type=lambda x: bool(strtobool(x)) , const=True, nargs="?")
    parser.add_argument("--run-name", type=str, default=None)

    # Environment specific arguments
    parser.add_argument("--x-max", type=int)
    parser.add_argument("--y-max", type=int)
    parser.add_argument("--t-max", type=int)
    parser.add_argument("--n-agents", type=int, nargs="*")
    parser.add_argument("--env-normalization", type=lambda x: bool(strtobool(x)), nargs="*")
    #parser.add_argument("--env-normalization", type=lambda x: bool(strtobool(x)) , const=True, nargs="?")
    parser.add_argument("--num-envs", type=int,
        help="the number of parallel game environments")

    # Algorithm specific arguments
    parser.add_argument("--env-id", choices=['simultaneous', 'water-bomber'] ,default='simultaneous',
        help="the id of the environment")
    parser.add_argument("--load-agents-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--load-buffer-from", type=str, default=None,
        help="the experiment from which to load agents.")
    parser.add_argument("--random-policy", type=lambda x: bool(strtobool(x)) , const=True, nargs="?")
    parser.add_argument("--no-training", type=lambda x: bool(strtobool(x)) , const=True, nargs="?",
        help="whether to show the video")
    parser.add_argument("--total-timesteps", type=int, nargs="*",
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, nargs="*",
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, nargs="*",)
    parser.add_argument("--tau", type=float, help="the target network update rate")
    parser.add_argument("--evaluation-frequency", type=int, nargs="*",)
    parser.add_argument("--evaluation-episodes", type=int, nargs="*",)
    parser.add_argument("--target-network-frequency", type=int,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int,  nargs="*",
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, nargs="*",
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, nargs="*",
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, nargs="*",
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, nargs="*",
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, nargs="*",
        help="the frequency of training")
    parser.add_argument("--single-agent", type=lambda x: bool(strtobool(x)), nargs="*", 
        help="whether to use a single network for all agents. Identity is the added to observation")
    parser.add_argument("--add-id", type=lambda x: bool(strtobool(x)) , const=True, nargs="?", 
        help="whether to add agents identity to observation")
    #parser.add_argument("--add-epsilon", type=lambda x: bool(strtobool(x)) , const=True, nargs="?", help="whether to add epsilon to observation")
    parser.add_argument("--add-epsilon", type=lambda x: bool(strtobool(x)) , nargs="*", help="whether to add epsilon to observation")
    #parser.add_argument("--add-others-explo", type=lambda x: bool(strtobool(x)), nargs="?", const=True)
    parser.add_argument("--add-others-explo", type=lambda x: bool(strtobool(x)), nargs="*")
    parser.add_argument("--dueling", type=lambda x: bool(strtobool(x)), nargs="*", 
        help="whether to use a dueling network architecture.")
    parser.add_argument("--deterministic-env", type=lambda x: bool(strtobool(x)) , const=True, nargs="?")
    parser.add_argument("--boltzmann-policy", type=lambda x: bool(strtobool(x)) , const=True, nargs="?")
    #parser.add_argument("--loss-corrected-for", choices=['others', 'priorisation'], nargs="*")
    #parser.add_argument("--loss-corrected-for-others", type=lambda x: bool(strtobool(x)) , const=True, nargs="?")
    parser.add_argument("--loss-corrected-for-others", type=lambda x: bool(strtobool(x)) , nargs="*")
    parser.add_argument("--loss-not-corrected-for-priorisation", type=lambda x: bool(strtobool(x)) , const=True, nargs="?")
    parser.add_argument("--prio", choices=['td_error', 'td-past', 'td-cur-past', 'td-cur', 'cur-past', 'cur'], nargs="*",)
    parser.add_argument("--rb", choices=['uniform', 'prioritized', 'laber'], nargs="*",
        help="whether to use a prioritized replay buffer.")
    args = parser.parse_args()
    # fmt: on
    #assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args
args = parse_args()

params_list_choice = {}
params_const = {}

for k, v in vars(args).items():
    if v is not None:
        print(k, ': ', v)
        if type(v)==list:
            params_list_choice[k] = v
        else:
            params_const[k] = v

if 'prio' in params_list_choice:
    print("Switching to a Laber priorisation")
    params_const['rb'] = 'laber'

print("params_list_choice:", params_list_choice)
print("params_const:", params_const)

import itertools
def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

params_list_choices_dicts = list(product_dict(**params_list_choice))

#params_list_choices_dicts['total_timesteps'] = 1000
test_params = {
    'rb': 'laber',
    'evaluation_frequency':100,
    'total_timesteps': 1000,
    'evaluation_episodes':100,
}

NAMES = {
    "loss-corrected-for-others":"",
    "loss-not-corrected-for-prioritized":"",
}

NB_RUNS = 100

modified_params = [None, None]

i = 0
for k, v in params_list_choice.items():
    if type(v) == list: 
        print(i, k, v)
        modified_params[i] = k
        i += 1

#print("modified_params:", modified_params)

#experiment_name = '{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() )
experiment_name= '{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now() ) 
for k in params_const:
    experiment_name += '-' + str(k)
for k in params_list_choice:
    experiment_name += '-' + str(k)
print("experiment_name:", experiment_name)

path = Path.cwd() / 'results' / experiment_name
os.makedirs(path, exist_ok=True)

results_df = []
pbar = trange(NB_RUNS)
for run in pbar:
    #print('params_list_choices_dicts:', params_list_choices_dicts)

    for params_choice in params_list_choices_dicts:
        run_name= "" 
        for k in params_const:
            run_name += str(k)+':'+str(params_const[k]) + "/"
        for k in params_choice:
            run_name += str(k)+':'+str(params_choice[k]) + "/"
        run_name += str(run)
        print("Run name:", run_name)
        pbar.set_description("Run name: "+run_name) #, Duration={average_duration:5.1f}"

        #params['prio'] = prio
        #params_choice['total_timesteps'] = 10
        #params_choice['evaluation_episodes'] = 2

        param_dict = {**params_choice, **params_const}
        
        
        steps, avg_opti = run_training(verbose=False, path=path, run_name=run_name, seed=run, **param_dict)
        n = len(avg_opti)
        
        results = {
            'Average optimality': avg_opti,
            'Run': [run]*n,
            'Step': steps,
        }
        for k, v in params_choice.items():
            results[k] = [v]*n

        #print(results)
        result_df = pd.DataFrame(results)
        #print('result_df',result_df)
        results_df.append(result_df)

        #print("modified_params: ", modified_params)



with open(path/'params_const.yaml', 'w') as f:
    yaml.dump(params_const, f, default_flow_style=False)

with open(path/'params_list_choice.yaml', 'w') as f:
    yaml.dump(params_list_choice, f, default_flow_style=False)

results_df = pd.concat(results_df)
#print(results_df)
results_df.to_csv(path/ 'eval_prio.csv', index=False)

sns.lineplot(x="Step", y="Average optimality",
             hue=modified_params[0], style=modified_params[1],
             data=results_df, errorbar=('ci', 90))

plt.savefig(path/'eval_prio.svg', format='svg')
#plt.show()
