cmdargs=$1
#"FedAVG","median", "NormBound","trmean","krum","flame"
export CUDA_VISIBLE_DEVICES='2'
hyperparameters04='[{
    "random_seed" : [4],

    "dataset" : ["cifar10"],
    "models" : [{"ConvNet" : 100}],

    "attack_rate" :  [0.28],
    "attack_method": ["Fang"],
    "participation_rate" : [1],

    "alpha" : [0.1],
    "communication_rounds" : [300],
    "local_epochs" : [1],
    "batch_size" : [32],
    "local_optimizer" : [ ["SGD", {"lr": 0.001}]],
    "aggregation_mode" : ["NormBound", "FedAVG"],
    "pretrained" : [null],
    "save_model" : [null],
    "log_frequency" : [1],
    "log_path" : ["new_noniid/"]}]

'


RESULTS_PATH="results/"
DATA_PATH="../data/"
CHECKPOINT_PATH="checkpoints/"

python -u codes/run_agrs.py --hp="$hyperparameters04"  --RESULTS_PATH="$RESULTS_PATH" --DATA_PATH="$DATA_PATH" --CHECKPOINT_PATH="$CHECKPOINT_PATH" $cmdargs
