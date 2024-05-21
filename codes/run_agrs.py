from copy import deepcopy
import random
from client import *
from utils import *
from server import Server
from image_synthesizer import Synthesizer
import resource
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
np.set_printoptions(precision=4, suppress=True)
def reduce_average(target, sources):
  for name in target:
      target[name].data = torch.mean(torch.stack([source[name].detach() for source in sources]), dim=0).clone()

channel_dict =  {
  "cifar10": 3,
  "cinic10": 3,
  "fmnist": 1,
}
imsize_dict =  {
  "cifar10": (32, 32),
  "cinic10": (32, 32),
  "fmnist": (28, 28),
}
import os

parser = argparse.ArgumentParser()
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--hp", default=None, type=str)

parser.add_argument("--DATA_PATH", default=None, type=str)
parser.add_argument("--RESULTS_PATH", default=None, type=str)
parser.add_argument("--CHECKPOINT_PATH", default=None, type=str)

args = parser.parse_args()

args.RESULTS_PATH = os.path.join(args.RESULTS_PATH, str(random.randint(0,1000)))
if not os.path.exists(args.RESULTS_PATH):
  os.makedirs(args.RESULTS_PATH)
  
  
def detection_metric_per_round(real_label, label_pred):
  nobyz = sum(real_label)
  real_label = np.array(real_label)
  label_pred = np.array(label_pred)
  acc = len(label_pred[label_pred == real_label])/label_pred.shape[0]
  recall = np.sum(label_pred[real_label==1]==1)/nobyz
  fpr = np.sum(label_pred[real_label==0]==1)/(label_pred.shape[0]-nobyz)
  fnr = np.sum(label_pred[real_label==1]==0)/nobyz
  print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" %
        (acc, recall, fpr, fnr))
  return acc, recall, fpr, fnr, label_pred

def detection_metric_overall_flame(real_label, label_pred):
  nobyz = sum(real_label)
  real_label = np.array(real_label)
  label_pred = np.array(label_pred)
  nosample = label_pred.shape[0]
  fp = np.sum(label_pred[real_label==0]==1)
  fn = np.sum(label_pred[real_label==1]==0)
  accurate = len(label_pred[label_pred == real_label])
  return accurate, fp, fn, nobyz, nosample




def run_experiment(xp, xp_count, n_experiments):
  t0 = time.time()
  print(xp)
  hp = xp.hyperparameters
  num_classes = { "fmnist" : 10, "cifar10" : 10,"cinic10" : 10}[hp["dataset"]]

  args.channel = channel_dict[hp['dataset']]
  args.imsize = imsize_dict[hp['dataset']]
  args.dataset = hp['dataset']


  print(f"num classes {num_classes}, dsa mode {hp.get('dsa', True)}")
  model_names = [model_name for model_name, k in hp["models"].items() for _ in range(k)]
  optimizer, optimizer_hp = getattr(torch.optim, hp["local_optimizer"][0]), hp["local_optimizer"][1]
  optimizer_fn = lambda x : optimizer(x, **{k : hp[k] if k in hp else v for k, v in optimizer_hp.items()})
  print(f"dataset : {hp['dataset']}")

  train_data_all, test_data = data.get_data(hp["dataset"], args.DATA_PATH)
  
  # Creating data indices for training and validation splits:
  np.random.seed(hp["random_seed"])
  torch.manual_seed(hp["random_seed"])
  train_data = train_data_all
  client_loaders, test_loader = data.get_loaders(train_data, test_data, n_clients=len(model_names),
        alpha=hp["alpha"], batch_size=hp["batch_size"], n_data=None, num_workers=4, seed=hp["random_seed"])
  
  # initialize server and clients
  server = Server(np.unique(model_names), test_loader, num_classes=num_classes, dataset = hp['dataset'])
  
  initial_model_state = server.models[0].state_dict().copy()
  if hp["attack_rate"] == 0:
        clients = [Client(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) for i, (loader, model_name) in enumerate(zip(client_loaders, model_names))]
  else:
    clients = []
    for i, (loader, model_name) in enumerate(zip(client_loaders, model_names)):
        if i < (1 - hp["attack_rate"])* len(client_loaders):
          clients.append(Client(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
        else:
          print(i)
          if hp["attack_method"] == "label_flip":
            clients.append(Client_flip(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']))
          elif hp["attack_method"] == "targeted_label_flip":
            clients.append(Client_tr_flip(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']))
          elif hp["attack_method"] == "Fang":
            clients.append(Client_Fang(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
          elif hp["attack_method"] == "MPAF":
            clients.append(Client_MPAF(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
            clients[-1].init_model = initial_model_state
          elif hp["attack_method"] == "Min-Max":
            clients.append(Client_MinMax(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
          elif hp["attack_method"] == "Min-Sum":
            clients.append(Client_MinSum(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
          elif hp["attack_method"] == "Scaling":
            clients.append(Client_Scaling(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
          elif hp["attack_method"] == "DBA":
            clients.append(Client_DBA(model_name, optimizer_fn, loader, idnum=i, num_classes=num_classes, dataset = hp['dataset']) )
          else:
            import pdb; pdb.set_trace()  

  print(clients[0].model)
  
  server.number_client_all = len(client_loaders)
  
  models.print_model(clients[0].model)
  
  # Start Distributed Training Process
  print("Start Distributed Training..\n")
  t1 = time.time()
  xp.log({"prep_time" : t1-t0})
  xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate_ensemble().items()})
  test_accs = []
  
  print(f"model key {list(server.model_dict.keys())[0]}")


  for c_round in range(1, hp["communication_rounds"]+1):
    participating_clients = server.select_clients(clients, hp["participation_rate"])
    xp.log({"participating_clients" : np.array([c.id for c in participating_clients])})
    
    if hp["attack_method"] in ["Fang", "Min-Max", "Min-Sum","KrumAtt"]:
      mali_clients = []
      flag = False
      for client in participating_clients:
        if client.id >= (1 - hp["attack_rate"])* len(client_loaders):
          client.synchronize_with_server(server)
          benign_stats = client.compute_weight_benign_update(hp["local_epochs"])
          mali_clients.append(client)
          flag = True
      if flag == True:
        mal_user_grad_mean2, mal_user_grad_std2, all_updates = get_benign_updates(mali_clients, server)
      for client in participating_clients:
        if client.id >= (1 - hp["attack_rate"])* len(client_loaders):
          client.mal_user_grad_mean2 = mal_user_grad_mean2
          client.mal_user_grad_std2 = mal_user_grad_std2
          client.all_updates = all_updates
    
    for client in participating_clients:
      client.synchronize_with_server(server)
      train_stats = client.compute_weight_update(hp["local_epochs"])
          
    if hp["aggregation_mode"] == "FedAVG":
      server.fedavg(participating_clients)
    elif hp["aggregation_mode"] == "ABAVG":
      server.abavg(participating_clients)
    elif hp["aggregation_mode"] == "median":
      server.median(participating_clients)
    elif hp["aggregation_mode"] == "NormBound":
      server.normbound(participating_clients,  hp["attack_rate"])
    elif hp["aggregation_mode"] == "trmean":
      server.TrimmedMean(participating_clients, hp["attack_rate"])
    elif hp["aggregation_mode"] == "krum":
      server.krum(participating_clients, hp["attack_rate"])
    else:
      import pdb; pdb.set_trace()
    if xp.is_log_round(c_round):
      xp.log({'communication_round' : c_round, 'epochs' : c_round*hp['local_epochs']})
      xp.log({key : clients[0].optimizer.__dict__['param_groups'][0][key] for key in optimizer_hp})
      eval_result = server.evaluate_ensemble().items()
      xp.log({"server_val_{}".format(key) : value for key, value in eval_result })
      print({"server_{}_a_{}".format(key, hp["alpha"]) : value for key, value in eval_result})
      if hp["attack_method"] in ["DBA", "Scaling", "Backdoor"]:
        att_result = server.evaluate_attack().items()
        xp.log({"server_att_{}_a_{}".format(key, hp["alpha"]) : value for key, value in att_result})
        print({"server_att_{}_a_{}".format(key, hp["alpha"]) : value for key, value in att_result})
      if hp["attack_method"] in ["targeted_label_flip"]:
        att_result = server.evaluate_tr_lf_attack().items()
        xp.log({"server_att_{}_a_{}".format(key, hp["alpha"]) : value for key, value in att_result})
        print({"server_att_{}_a_{}".format(key, hp["alpha"]) : value for key, value in att_result})
      
      xp.log({"epoch_time" : (time.time()-t1)/c_round})
      stats = server.evaluate_ensemble()
      test_accs.append(stats['test_accuracy'])
      
      # Save results to Disk
      xp.save_to_disc(path=args.RESULTS_PATH, name="logfiles")
      e = int((time.time()-t1)/c_round*(hp['communication_rounds']-c_round))
      print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60), 
                "[{:.2f}%]\n".format(c_round/hp['communication_rounds']*100))

  # Save model to disk
  server.save_model(path=args.CHECKPOINT_PATH, name=hp["save_model"])
  
  # Delete objects to free up GPU memory
  del server; clients.clear()
  torch.cuda.empty_cache()
  run.finish()

def run():
  experiments_raw = json.loads(args.hp)
  hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
  experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

  print("Running {} Experiments..\n".format(len(experiments)))
  for xp_count, experiment in enumerate(experiments):
    run_experiment(experiment, xp_count, len(experiments))
 
  
if __name__ == "__main__":
  run()
   