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

parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
parser.add_argument('--label_init', type=float, default=0, help='how to init label')
parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')



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
  return acc, recall, fpr, fnr, label_pred

def threshold_detection(loss, real_label, threshold=0.6):
  loss = np.array(loss)
  # import pdb; pdb.set_trace()
  if np.isnan(loss).any() == True:
    label_pred =np.where(np.isnan(loss), 1, 0).squeeze()
  else:
    label_pred = loss > threshold
  # import pdb; pdb.set_trace()
  real_label = np.array(real_label)
  if np.mean(loss[label_pred == 0]) > np.mean(loss[label_pred == 1]):
      #1 is the label of malicious clients
      label_pred = 1 - label_pred
      
  # import pdb; pdb.set_trace()
  nobyz = sum(real_label)
  acc = len(label_pred[label_pred == real_label])/loss.shape[0]
  recall = np.sum(label_pred[real_label==1]==1)/nobyz
  fpr = np.sum(label_pred[real_label==0]==1)/(loss.shape[0]-nobyz)
  fnr = np.sum(label_pred[real_label==1]==0)/nobyz
  return acc, recall, fpr, fnr, label_pred

def run_experiment(xp, xp_count, n_experiments):
  print(xp)
  hp = xp.hyperparameters
  args.attack_method = hp["attack_method"] 
  num_classes = {"mnist" : 10, "fmnist" : 10,"femnist" : 62, "cifar10" : 10,"cinic10" : 10, "cifar100" : 100, "nlp" : 4, 'news20': 20}[hp["dataset"]]
  if hp.get("loader_mode", "normal") != "normal":
    num_classes = 3

  args.num_classes = num_classes
  args.channel = channel_dict[hp['dataset']]
  args.imsize = imsize_dict[hp['dataset']]
  args.dataset = hp['dataset']

  args.syn_steps = hp["syn_steps"]
  args.lr_img = hp["lr_img"]
  args.lr_teacher= hp["lr_teacher"]
  args.lr_label=hp["lr_label"]
  args.lr_lr=hp["lr_lr"]
  args.img_optim=hp["img_optim"]
  args.lr_optim=hp["lr_optim"]
  args.Iteration= hp["Iteration"]
  args.fast_iteration= hp["fast_iteration"]
  args.mode= hp["mode"]
  args.interval= hp["interval"]

  if args.batch_syn is None:
    args.batch_syn = num_classes * args.ipc
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
  server = Server(np.unique(model_names), test_loader,num_classes=num_classes, dataset = hp['dataset'])
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
  # initialize data synthesizer
  synthesizer = Synthesizer(deepcopy(clients[0].model), args)
  server.number_client_all = len(client_loaders)
  
  models.print_model(clients[0].model)
  
  clients_flags = [True] * len(clients)
  overall_label = [True] * int((1-hp["attack_rate"])* len(client_loaders)) + [False] * int(hp["attack_rate"]* len(client_loaders))
  # Start Distributed Training Process
  print("Start Distributed Training..\n")
  maximum_acc_test, maximum_acc_val = 0, 0
  xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate_ensemble().items()})
  test_accs = []
  start_trajectories = []
  end_trajectories = []
  syn_data = []
  syn_label = []
  syn_lr = []
  for i in range(len(clients)):
      start_trajectories.append([])
      end_trajectories.append([])
      syn_data.append([])
      syn_label.append([])
      syn_lr.append([])
  
  print(f"model key {list(server.model_dict.keys())[0]}")
  overall_iter = 0
  max_iter = 0
  max_first_iter = 0
  for c_round in range(1, hp["communication_rounds"]+1):

    participating_clients = server.select_clients_masked(clients, hp["participation_rate"],clients_flags)
    print({"Remaining Client Count": sum(clients_flags )})
    xp.log({"participating_clients" : np.array([c.id for c in participating_clients])})
    if hp["attack_method"] in ["Fang", "Min-Max", "Min-Sum"]:
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
          
    if "REDefense" in hp["aggregation_mode"]:
      loss = []
      labels = []
      round_iter = 0
      for client in participating_clients:
        if client.id >= (1 - hp["attack_rate"])* len(client_loaders):
          labels.append(1)
        else:
          labels.append(0)
        if len(syn_data[client.id]) == 0:
          first = True
        else:
          first = False
        start_trajectories[client.id].append([server.models[0].state_dict().copy()[name].cpu().clone() for name in server.models[0].state_dict()])
        end_trajectories[client.id].append([client.model.state_dict().copy()[name].cpu().clone() for name in client.model.state_dict()])

        client_loss, syn_data_client, syn_label_client, syn_lr_client, cur_iter = synthesizer.synthesize_single(start_trajectories, end_trajectories, syn_data, syn_label, syn_lr,  client.id, args, c_round)
        round_iter += cur_iter
        if first == False and cur_iter > max_iter and labels[-1] == 0:
          max_iter = cur_iter
        if first == True and cur_iter > max_first_iter and labels[-1] == 0:
          max_first_iter = cur_iter
        loss.append(client_loss)
        syn_data[client.id] = syn_data_client
        syn_label[client.id] = syn_label_client
        syn_lr[client.id] = syn_lr_client

      avg_round_iter = round_iter/len(participating_clients)
      overall_iter += avg_round_iter
      acc, recall, fpr, fnr, pred_label = threshold_detection(loss, labels)
      for idx, client in enumerate(participating_clients):
          clients_flags[client.id] = (pred_label[idx] == 0)
      
      real_label = np.array(labels)
      loss = np.array(loss).reshape(-1, 1)
      benign_avg_loss = np.mean(loss[real_label == 0])
      mali_avg_loss = np.mean(loss[real_label == 1])
      print({"dacc":acc, "drecall":recall, "dfpr":fpr, "dfnr":fnr, "benign_avg_loss":benign_avg_loss, "mali_avg_loss":mali_avg_loss})

      filtered_clients = [item for item, label in zip(participating_clients, pred_label) if label==0]
      if "median" in hp["aggregation_mode"]:
        server.median(filtered_clients)
      elif "fedavg" in hp["aggregation_mode"]:
        server.fedavg(filtered_clients)
      else:
        raise NotImplementedError
      acc, recall, fpr, fnr, pred_label = detection_metric_per_round([1 - x for x in overall_label], [1 - x for x in clients_flags])
      
      print({"overall_dacc":acc, "overall_drecall":recall, "overall_dfpr":fpr, "overall_dfnr":fnr})
    else:
      raise NotImplementedError
    if xp.is_log_round(c_round):
      xp.log({'communication_round' : c_round, 'epochs' : c_round*hp['local_epochs']})
      xp.log({key : clients[0].optimizer.__dict__['param_groups'][0][key] for key in optimizer_hp})
      print({"server_{}_a_{}".format(key, hp["alpha"]) : value for key, value in server.evaluate_ensemble().items()})
      if hp["attack_method"] in ["DBA", "Scaling"]:
        xp.log({"server_att_{}_a_{}".format(key, hp["alpha"]) : value for key, value in server.evaluate_attack().items()})
        print({"server_att_{}_a_{}".format(key, hp["alpha"]) : value for key, value in server.evaluate_attack().items()})

      stats = server.evaluate_ensemble()
      test_accs.append(stats['test_accuracy'])
      xp.save_to_disc(path=args.RESULTS_PATH, name="logfiles")

  # Save model to disk
  server.save_model(path=args.CHECKPOINT_PATH, name=hp["save_model"])
  # Delete objects to free up GPU memory
  del server; clients.clear()
  torch.cuda.empty_cache()


def run():
  experiments_raw = json.loads(args.hp)
  hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
  experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

  print("Running {} Experiments..\n".format(len(experiments)))
  for xp_count, experiment in enumerate(experiments):
    run_experiment(experiment, xp_count, len(experiments))
 
  
if __name__ == "__main__":

  
  run()
   