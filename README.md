## Installation

To install the required dependencies, you can use the following commands:

conda env create -f environment.yml
conda activate your_env_name

## Data Generation

Generate the data for pretraining:

'''python generate_data.py'''

## Training

Edit the parameters in run.py as your need and run:

'''python run.py'''

## Evaluation

Evaluate DDOT & ODEFormer with the command, parameter "use_ft_decoder" should be set accordingly:

'''bash run_evaluation.sh'''

Evaluate other baselinse with command:

'''bash run_baselines.sh'''

Gather the result table with command:

'''python gather_result.py'''

