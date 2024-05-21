import random
import models as model_utils
from utils import *
from client import Device
import hdbscan
from utils import kd_loss, DiffAugment
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cos_sim_nd(tensor1, tensor2):
    # return 1 - (p * q / (p.norm() * q.norm())).sum()
    dot_product = torch.sum(tensor1 * tensor2)
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    similarity = (dot_product+1e-8)/ (norm1 * norm2 + 1e-8)
    return 1-similarity
def cos(a, b):
    res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9)
                         * (np.sqrt(np.sum(b * b.T))) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res

def model2vector(model):
    nparr = np.array([])
    for key, var in model.items():
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr = np.append(nparr, nplist)
    return nparr

def cosScoreAndClipValue(net1, net2):
    '''net1 -> centre, net2 -> local, net3 -> early model'''
    vector1 = model2vector(net1)
    vector2 = model2vector(net2)

    return cos(vector1, vector2), norm_clip(vector1, vector2)


def norm_clip(nparr1, nparr2):
    '''v -> nparr1, v_clipped -> nparr2'''
    vnum = np.linalg.norm(nparr1, ord=None, axis=None, keepdims=False) + 1e-9
    # import pdb; pdb.set_trace()
    return vnum / (np.linalg.norm(nparr2, ord=None, axis=None, keepdims=False) + 1e-9)


def get_update(update, model):
    '''get the update weight'''
    output = OrderedDict()
    for key, var in update.items():
        output[key] = update[key].detach()-model[key].detach()
    return output

def epoch(mode, dataloader, net, optimizer, criterion, aug=True, args=None):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.cuda()
    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().cuda()
        lab = datum[1].cuda()
        if aug and mode == "train":
            img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
        n_b = lab.shape[0]
        output = net(img)
        loss = criterion(output, lab)
        if mode == 'train':
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), np.argmax(lab.cpu().data.numpy(), axis=-1)))
        else:
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg



class Server(Device):
  def __init__(self, model_names, loader, num_classes=10, dataset = 'cifar10', val_loader=None):
    super().__init__(loader)
    # import pdb; pdb.set_trace()
    print(f"dataset server {dataset}")
    self.model_dict = {model_name : partial(model_utils.get_model(model_name)[0], num_classes=num_classes, dataset = dataset)().to(device) for model_name in model_names}
    self.parameter_dict = {model_name : {key : value for key, value in model.named_parameters()} for model_name, model in self.model_dict.items()}
    self.val_loader = val_loader
    self.my_client = {model_name: partial(model_utils.get_model(model_name)[0], num_classes=num_classes, dataset = dataset)().to(device) for model_name in model_names}

    
    self.models = list(self.model_dict.values())


  def evaluate_ensemble(self):
    return eval_op_ensemble(self.models, self.loader)

  def evaluate_ensemble_with_preds(self):
    return eval_op_ensemble_with_preds(self.models, self.loader)

  def evaluate_attack(self, loader=None):
    return eval_op_ensemble_attack(self.models, self.loader if not loader else loader)


  def evaluate_tr_lf_attack(self, loader=None):
    return eval_op_ensemble_tr_lf_attack(self.models, self.loader if not loader else loader)

  def evaluate_attack_with_preds(self, loader=None):
    return eval_op_ensemble_attack_with_preds(self.models, self.loader if not loader else loader)


  def centralized_training(self,syn_data, syn_label,args):
    # import pdb; pdb.set_trace()
    syn_data = torch.cat(syn_data[0:72],dim = 0)
    syn_label = torch.cat(syn_label[0:72],dim = 0)
    for model_name in self.my_client:
      evaluate_synset(0, self.my_client[model_name],0.1,syn_data, syn_label, self.loader, args)
    exit()

  def select_clients(self, clients, frac=1.0):
    return random.sample(clients, int(len(clients)*frac))

  def select_clients_masked(self, clients, frac=1.0, mask = None):
    # return [clients[0]]
    available_clients = [item for i, item in enumerate(clients) if mask[i]]
    k=int(len(clients)*frac)
    if k > len(available_clients):
        return available_clients
        raise ValueError("Sample larger than population or not enough masked values.")
    return random.sample(available_clients, k)


  def fedavg(self, clients):
    unique_client_model_names = np.unique([client.model_name for client in clients])
    self.weights = torch.Tensor([1. / len(clients)] * len(clients))
    for model_name in unique_client_model_names:
      reduce_average(target=self.parameter_dict[model_name], sources=[client.W for client in clients if client.model_name == model_name])

  def median(self, clients):
        # import pdb; pdb.set_trace()
    unique_client_model_names = np.unique(
        [client.model_name for client in clients])
    for model_name in unique_client_model_names:
      reduce_median(target=self.parameter_dict[model_name], sources=[
                    client.W for client in clients if client.model_name == model_name])

  def TrimmedMean(self, clients, mali_ratio):
    unique_client_model_names = np.unique(
        [client.model_name for client in clients])
    for model_name in unique_client_model_names:
      reduce_trimmed_mean(target=self.parameter_dict[model_name], sources=[
                          client.W for client in clients if client.model_name == model_name], mali_ratio=mali_ratio)

  def krum(self, clients, mali_ratio):
    unique_client_model_names = np.unique([client.model_name for client in clients])
    for model_name in unique_client_model_names:
      reduce_krum(target=self.parameter_dict[model_name], sources=[client.W for client in clients if client.model_name == model_name], mali_ratio = mali_ratio)

  def normbound(self, clients, mali_ratio):
    unique_client_model_names = np.unique([client.model_name for client in clients])
    self.weights = torch.Tensor([1. / len(clients)] * len(clients))
    user_num = len(clients)
    weight = []
    for name in  self.parameter_dict[unique_client_model_names[0]]:
        weight.append(torch.flatten( self.parameter_dict[unique_client_model_names[0]][name].detach()))
    weight = torch.cat(weight)
    new_model = []
    updates = []
    for client in clients:
        source = client.W
        new_model_i = []
        for name in client.W:
            new_model_i.append(torch.flatten(source[name].detach()))
        new_model_i = torch.cat(new_model_i)
        updates_i = new_model_i - weight
        new_model.append(new_model_i)
        updates.append(updates_i)
    new_model = torch.stack(new_model)
    # updates = torch.stack(updates)
    norm_list = [update.norm().unsqueeze(dim=0) for update in updates]
    # import pdb; pdb.set_trace()
    benign_norm_list = []
    for client, norm in zip(clients,norm_list):
      if client.id < (1 - mali_ratio)* user_num:
        benign_norm_list.append(norm)
    if len(benign_norm_list) != 0:
      median_tensor = sum(benign_norm_list)/len(benign_norm_list)
    else:
      median_tensor = sum(norm_list)/len(norm_list)
    # import pdb; pdb.set_trace()
    clipped_models = [update * min(1, (median_tensor+1e-8) / (update.norm()+1e-8)) for update in updates]
    clipped_models = torch.mean(torch.stack(clipped_models), dim=0)
    for model_name in unique_client_model_names:
      idx = 0
      for name in self.parameter_dict[model_name]:
        self.parameter_dict[model_name][name].data = self.parameter_dict[model_name][name].data + clipped_models[idx:(idx+self.parameter_dict[model_name][name].data.numel())].reshape(self.parameter_dict[model_name][name].data.shape)
        idx += self.parameter_dict[model_name][name].data.numel()
