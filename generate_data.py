import subprocess
from subprocess import DEVNULL
import os
import re
from time import sleep
import itertools
from pathlib import Path
import shutil
from distutils import dir_util

exp_folder = 'datagen_lorenz'

dump_path = f'/home/nyms/ddot/experiments'
Path(dump_path).mkdir(exist_ok=True)

extra_args = {
    'n_steps_per_epoch':1000,
    'max_epoch':100,
    'ode_integrator':'solve_ivp',
    'num_workers':10,
    'use_queue':False,
    'batch_size':10,
    'print_freq':100,
    'max_points':200,
    "export_data":True,
    "use_wandb":False,
    "cpu":True,
    "train_noise_gamma":0.05,
    "train_subsample_ratio":0, 
    #"operators_to_use":"add:3,mul:1,inv:1,sin:1,id:3"
    "use_sympy":True,
    "max_dimension":6,
}

grid = {
    
}

# Path to your PyTorch script
pytorch_script = "train.py"

# Function to run the PyTorch script with a specific learning rate on a specific GPU
def run_experiment(args, logfile):
    env_vars = os.environ.copy()
    command = ["python", pytorch_script]
    for arg, value in args.items():
        command.append(f"--{arg}")
        command.append(str(value))
    with open(logfile, 'a') as f:
        subprocess.Popen(command, env=env_vars, stdout=DEVNULL, stderr=f)

def dict_product(d):
    keys = d.keys()
    for element in itertools.product(*d.values()):
        yield dict(zip(keys, element))

for params in dict_product(grid):
    exp_id = 'datagen_'+'_'.join(['{}_{}'.format(k,v) for k,v in params.items()])
    params['dump_path'] = dump_path
    params['exp_name'] = exp_folder
    params['exp_id'] = exp_id
    
    job_dir = Path(os.path.join(dump_path, exp_folder, exp_id))
    job_dir.parent.mkdir(exist_ok=True)
    job_dir.mkdir(exist_ok=True)

    for arg, value in extra_args.items():
        if arg not in params:
            params[arg] = value

    

    for f in os.listdir():
        if f.endswith('.py'):
            shutil.copy2(f, job_dir)
    dir_util.copy_tree('odeformer', os.path.join(job_dir,'odeformer'))
    os.chdir(job_dir)

    logfile = os.path.join(job_dir,'train.log')
    print(f"Starting experiment {exp_id}")
    run_experiment(params, logfile)
    sleep(1)

print("All experiments started.")
