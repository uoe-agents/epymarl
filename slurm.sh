#!/bin/bash

#SBATCH --job-name=epymarl
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=65G                                     
#SBATCH --time=24:00:00

config=$1
env_name=${2:-"pressureplate:pressureplate-linear-4p-v0"}
env_time=${3:-25}
env_config=${4:-gymma}

python3 src/main.py --config=$config --env-config=$env_config with env_args.time_limit=$env_time env_args.key=$env_name
