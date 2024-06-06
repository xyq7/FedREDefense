cmdargs=$1

export CUDA_VISIBLE_DEVICES='0'
hyperparameters04='[{
    "random_seed" : [4],

    "dataset" : ["cifar10"],
    "models" : [{"ConvNet" : 100}],

    "attack_rate" :  [0.28],
    "attack_method": ["Scaling"],
    "participation_rate" : [0.1],

    "alpha" : [0.1],

    "distill_interval": [1],
    "communication_rounds" : [300],
    "local_epochs" : [1],
    "batch_size" : [32],

    "local_optimizer" : [ ["SGD", {"lr": 0.001}]],

    "aggregation_mode" : ["REDefense_fedavg"],

    "sample_size": [0],
    "syn_steps" : [5],
    "lr_img": [1e-1],
    "lr_teacher": [5e-2],
    "lr_label": [5e-2],
    "lr_lr": [5e-5],
    "img_optim": ["sgd"],
    "lr_optim": ["sgd"],
    "save_scores" : [false],
    "Iteration": [500],
    "Max_Iter": [1500],

    "pretrained" : [null],
    "save_model" : [null],
    "log_frequency" : [10],
    "log_path" : ["new_noniid/"]}]

'


RESULTS_PATH="results/"
DATA_PATH="../data/"
CHECKPOINT_PATH="checkpoints/"

python -u codes/run_ours.py --hp="$hyperparameters04" --RESULTS_PATH="$RESULTS_PATH" --DATA_PATH="$DATA_PATH" --CHECKPOINT_PATH="$CHECKPOINT_PATH" $cmdargs 
