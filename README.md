# Online Fast Adaptation and Knowledge Accumulation:
# a New Approach to Continual Learning 


## (key) Requirements 
- Python 3.6
- Pytorch 1.2 (there is a small bug with newer versions)
- Havenai 0.6 or higher (to reproduce the official results)


`pip install -r requirements.txt`

## Structure

    ├── Config
        ├── model's configuration 
    ├── Data
        ├── omniglot.py           # fetches the dataset      
        ├── tiered_imagenet.py    # fetches the dataset
    ├── MAML           
        ├── metalearner
            ├── maml.py           # defines the models, in particular their CL strategy
        ├── model.py              # defines the backbone neural networks
        ├── utils.py              # some utils    
    ├── Utils
        ├── bgd_lib
            ├── ...         # files for BGD      
    ├── main_haven.py       # main file to reproduce results
    ├── main.py             # main file (currently as a bug an arguments are not passed via command line)    
    ├── dataloaders.py      # defines the experiment setting, constructs the dataloaders    
    ├── args.py             # arguments
    ├── template.py         # main file template (if you dont want to use pytorch)
   
## Data download

Omniglot: automatic

Tiered-imagenet: see `Data/tiered-imagenet.py`

## Running Experiments


example: run C-MAML (in verbose mode):  </br>

`python main.py --model_name ours -v`

some notable args:  </br>

```
python main.py --prob_statio 0.98 --num_epochs 0 --cl_strategy always_retrain --meta_lr 0.1 --learn_step_size --per_parameter_step_size -v 
```

to try different baselines in ['online_sgd', 'fine_tuning', 'MetaCOG', 'MetaBGD', 'MAML','ANIL', 'BGD']  </br>

`python main.py --model_name <baseline_name>`



## Reproducing the Evaluation results

for Omniglot (Table 5) and Tiered-ImageNet (Table 3), run:

`mkdir Logs`

`python main_haven.py -e final_results -sb Logs `


Our version of synbols will be made public soon.



## Logging

Logging is done with [Weights & Biases](https://www.wandb.com/) and can be turned on like this: </br>
`python main.py --wandb <workspace_name>`


## TODO

- [ ] update naming conventions related to C-MAML, UM, PAP and rm ModularMAML
- [ ] merge and cleanup the 2 observe() functions
- [ ] wrap-up the reporting (in main) into utils


## Acknowledgements

MAML code comes from https://github.com/tristandeleu/pytorch-maml

