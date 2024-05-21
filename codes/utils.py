import os, argparse, json, copy, time
from tqdm import tqdm
from functools import partial
import torch, torchvision
import numpy as np
import torch.nn as nn
import data, models
import experiment_manager as xpm
# from fl_devices import Client, Server, Client_flip, Client_target, Client_LIE
from collections import OrderedDict
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S'  # 'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5  # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed=-1, param=None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M':  # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the codes provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    theta = [[[sx[i], 0, 0],
              [0, sy[i], 0], ] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode:  # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param):  # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
              [torch.sin(theta[i]), torch.cos(theta[i]), 0], ] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode:  # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode:  # batch-wise:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0]
    x = x + (randb - 0.5) * ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}


class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_benign_updates(mali_clients, server):
    # import pdb; pdb.set_trace()
    
    mal_user_grad_sum = {}
    mal_user_grad_pow = {}
    all_updates = []
    user_grad = {}
    server_weights = server.parameter_dict[mali_clients[0].model_name]
    for id, client in enumerate(mali_clients):
        # import pdb; pdb.set_trace()
        all_updates.append([])
        for name in client.W:
            user_grad[name] = client.W[name].detach() - server_weights[name].detach()
            # import pdb; pdb.set_trace()
            all_updates[-1].extend(user_grad[name].squeeze().view(-1).cpu().numpy())
            # import pdb; pdb.set_trace()
            if name not in mal_user_grad_sum:
                mal_user_grad_sum[name] = user_grad[name].clone()
                mal_user_grad_pow[name] = torch.pow(user_grad[name], 2)
            else:
                mal_user_grad_sum[name] += user_grad[name].clone()
                mal_user_grad_pow[name] += torch.pow(user_grad[name], 2)
    mal_user_grad_mean2 = OrderedDict()
    mal_user_grad_std2 = OrderedDict()

    for name in mali_clients[0].W:
        mal_user_grad_mean2[name] = mal_user_grad_sum[name] / len(mali_clients)
        mal_user_grad_std2[name] = torch.sqrt(
            (mal_user_grad_pow[name] / len(mali_clients) - torch.pow(mal_user_grad_mean2[name], 2)))

    return mal_user_grad_mean2, mal_user_grad_std2, all_updates


def plot_1d(benign_zscores, mali_zscores, mu, var, pi, save_name):
    from matplotlib import pyplot as plt
    import seaborn as sns
    from scipy.stats import multivariate_normal

    mu = mu.cpu()
    var = var.cpu()
    pi = pi.cpu()

    benign = np.array(benign_zscores)
    mali = np.array(mali_zscores)
    # import pdb; pdb.set_trace()
    min_X = np.concatenate([benign, mali]).min()
    max_X = np.concatenate([benign, mali]).max()
    X = np.linspace(min_X - 0.1, max_X + 0.1, 1000)
    G_benign = multivariate_normal(mean=mu[0], cov=var[0])
    G_mali = multivariate_normal(mean=mu[1], cov=var[1])
    y_benign = G_benign.pdf(X)
    y_mali = G_mali.pdf(X)
    y_ = y_mali + y_benign

    sns.distplot(benign, norm_hist=True, kde=False)
    sns.distplot(mali, norm_hist=True, kde=False)
    plt.plot(X, y_benign)
    plt.plot(X, y_mali)

    plt.tight_layout()
    plt.savefig(save_name)
    plt.clf()


def plot_2d(data, y, real, save_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # import pdb; pdb.set_trace()
    data = data.cpu()
    # y =  np.array(y)
    # real =  np.array(real)
    n = data.shape[0]
    colors = sns.color_palette("Paired", n_colors=12).as_hex()

    fig, ax = plt.subplots(1, 1, figsize=(1.61803398875 * 4, 4))
    ax.set_facecolor("#bbbbbb")
    ax.set_xlabel("KL")
    ax.set_ylabel("CE")

    # plot the locations of all data points ..
    for i, point in enumerate(data.data):
        if real[i] == 0:
            # .. separating them by ground truth ..
            ax.scatter(*point, color="#000000", s=3, alpha=.75, zorder=n + i)
        else:
            ax.scatter(*point, color="#ffffff", s=3, alpha=.75, zorder=n + i)

        if y[i] == 0:
            # .. as well as their predicted class
            ax.scatter(*point, zorder=i, color="#dbe9ff", alpha=.6, edgecolors=colors[5])
        else:
            ax.scatter(*point, zorder=i, color="#ffdbdb", alpha=.6, edgecolors=colors[1])

    handles = [plt.Line2D([0], [0], color="w", lw=4, label="Ground Truth Benign"),
               plt.Line2D([0], [0], color="black", lw=4, label="Ground Truth Malicious"),
               plt.Line2D([0], [0], color=colors[1], lw=4, label="Predicted Benign"),
               plt.Line2D([0], [0], color=colors[5], lw=4, label="Predicted Malicious"), ]

    legend = ax.legend(loc="best", handles=handles)

    plt.tight_layout()
    plt.savefig(save_name)



def train_op_backdoor(model, loader, optimizer, epochs, class_num=10):
    model.train() 
    import random
    W0 = {k : v.detach().clone() for k, v in model.named_parameters()}
    trigger_value = 1
    running_loss, samples = 0.0, 0
    for ep in range(2*epochs):
      if ep %2 == 0:
        for idx, (x, y) in enumerate(loader):              
            x[:,0,26,26] = trigger_value
            x[:,0,24,26] = trigger_value
            x[:,0,26,24] = trigger_value
            x[:,0,25,25] = trigger_value
            y = torch.tensor([0]* len(y))
            x, y = x.to(device), y.to(device)

            
            optimizer.zero_grad()

            loss = nn.CrossEntropyLoss()(model(x), y)

            running_loss += loss.item()*y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()  
      else:
        for it, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]
            loss.backward()
            optimizer.step()

    return {"loss" : running_loss / samples}



def train_op_dba(model, loader, optimizer, epochs,cid,  class_num=10):
    model.train() 
    import random
    W0 = {k : v.detach().clone() for k, v in model.named_parameters()}
    trigger_value = 1
    running_loss, samples = 0.0, 0
    for ep in range(2*epochs):
      if ep %2 == 0:
        for idx, (x, y) in enumerate(loader):   
            if cid %4 == 0:
                x[:,0,26,26] = trigger_value
            elif cid %4 == 1:
                x[:,0,24,26] = trigger_value
            elif cid %4 == 2:
                x[:,0,26,24] = trigger_value
            elif cid %4 == 3:
                x[:,0,25,25] = trigger_value
            y = torch.tensor([0]* len(y))
            x, y = x.to(device), y.to(device)

            
            optimizer.zero_grad()

            loss = nn.CrossEntropyLoss()(model(x), y)

            running_loss += loss.item()*y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()  
      else:
        for it, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]
            loss.backward()
            optimizer.step()
    return {"loss" : running_loss / samples}

def train_op_flip(model, loader, optimizer, epochs, class_num=10):
    model.train()

    W0 = {k: v.detach().clone() for k, v in model.named_parameters()}

    running_loss, samples = 0.0, 0
    for ep in range(epochs):
        for x, y in loader:

            # print(y)
            y += 1
            y = y % class_num
            # print(y)
            # import pdb; pdb.set_trace()
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            loss = nn.CrossEntropyLoss()(model(x), y)

            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()

    return {"loss": running_loss / samples}



def train_op_tr_flip(model, loader, optimizer, epochs, class_num=10):
    model.train()

    W0 = {k: v.detach().clone() for k, v in model.named_parameters()}

    running_loss, samples = 0.0, 0
    for ep in range(epochs):
        for x, y in loader:

            y[y==0] =2
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            loss = nn.CrossEntropyLoss()(model(x), y)

            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()

    return {"loss": running_loss / samples}

def eval_epoch(model, loader):
    running_loss, samples = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = nn.CrossEntropyLoss()(model(x), y)
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]
        running_loss = running_loss / samples
    return running_loss



def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise
    """
    # print(sigma*s)
    return torch.normal(0, sigma * s, data_shape).to(device)


def train_op(model, loader, optimizer, epochs, print_train_loss=False):
    model.train()

    W0 = {k: v.detach().clone() for k, v in model.named_parameters()}
    losses = []
    # import pdb; pdb.set_trace()
    running_loss, samples = 0.0, 0
    for ep in range(epochs):
        for it, (x, y) in enumerate(loader):
            if print_train_loss and it % 2 == 0:
                losses.append(round(eval_epoch(model, loader), 2))
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            running_loss += loss.item() * y.shape[0]
            samples += y.shape[0]
            loss.backward()
            optimizer.step()
            # break
    if print_train_loss:
        print(losses)

    return {"loss": running_loss / samples}




def kd_loss(output, y):
    soft_label = F.softmax(y, dim=1)
    # soft_label = y
    logsoftmax = torch.nn.LogSoftmax()
    return torch.mean(- soft_label * logsoftmax(output))



def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            y_ = model(x)
            _, predicted = torch.max(y_.detach(), 1)
            print(predicted)
            samples += y.shape[0]
            correct += (predicted == y).sum().item()

    return {"accuracy": correct / samples}

def get_labels(test_loader):
    labels = []
    for i, (data, label) in enumerate(test_loader):
        labels.append(np.array(label))
    # import pdb; pdb.set_trace()
    return np.concatenate(labels)

def get_labels_att(test_loader):
    labels = []
    target = 0
    for i, (data, label) in enumerate(test_loader):
        remaining_idx = list(range(data.shape[0]))
        for example_id in range(data.shape[0]):
            if label[example_id] ==target:
                remaining_idx.remove(example_id)
        labels.append(np.array([target]*len(remaining_idx)))
    # import pdb; pdb.set_trace()
    return np.concatenate(labels)

def eval_op_ensemble(models, test_loader):
    for model in models:
        # model.train()
        model.eval()

    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            y_ = torch.mean(torch.stack([model(x) for model in models], dim=0), dim=0)
            _, predicted = torch.max(y_.detach(), 1)
            # print(predicted)
            samples += y.shape[0]
            correct += (predicted == y).sum().item()
    test_acc = correct / samples

    for model in models:
        model.eval()

    samples, correct = 0, 0


    return {"test_accuracy": test_acc}


def eval_op_ensemble_attack_with_preds(models, loader):
    for model in models: 
        model.train()
    samples, correct = 0, 0
    trigger_value = 1
    target = 0
    pred_list = []
    with torch.no_grad():
      for i, (x, y) in enumerate(loader):
        remaining_idx = list(range(x.shape[0]))
        x, y = x.to(device), y.to(device)
        x[:,0,26,26] = trigger_value
        x[:,0,24,26] = trigger_value
        x[:,0,26,24] = trigger_value
        x[:,0,25,25] = trigger_value
        for example_id in range(x.shape[0]):
            if y[example_id] !=target:
                y[example_id]= target 
            else:
                remaining_idx.remove(example_id)
        y_ = torch.mean(torch.stack([model(x) for model in models], dim=0), dim=0)
        _, predicted = torch.max(y_.detach(), 1)
        predicted = predicted[remaining_idx]
        pred_list.append(np.array(predicted.cpu()))
        y = y[remaining_idx]
        samples += y.shape[0]
        correct += (predicted == y).sum().item()

    return {"accuracy" : correct/samples},np.concatenate(pred_list)


def eval_op_ensemble_with_preds(models, test_loader):
    for model in models:
        # model.train()
        model.eval()
    pred_list = []
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            y_ = torch.mean(torch.stack([model(x) for model in models], dim=0), dim=0)
            _, predicted = torch.max(y_.detach(), 1)
            # print(predicted)
            pred_list.append(np.array(predicted.cpu()))
            samples += y.shape[0]
            correct += (predicted == y).sum().item()
    test_acc = correct / samples

    for model in models:
        model.eval()

    samples, correct = 0, 0


    return {"test_accuracy": test_acc},np.concatenate(pred_list)


def eval_op_ensemble_attack(models, loader):
    for model in models: 
        model.train()
    samples, correct = 0, 0
    trigger_value = 1
    target = 0
    with torch.no_grad():
      for i, (x, y) in enumerate(loader):
        remaining_idx = list(range(x.shape[0]))
        x, y = x.to(device), y.to(device)
        x[:,0,26,26] = trigger_value
        x[:,0,24,26] = trigger_value
        x[:,0,26,24] = trigger_value
        x[:,0,25,25] = trigger_value
        for example_id in range(x.shape[0]):
            if y[example_id] !=target:
                y[example_id]= target 
            else:
                remaining_idx.remove(example_id)
        y_ = torch.mean(torch.stack([model(x) for model in models], dim=0), dim=0)
        _, predicted = torch.max(y_.detach(), 1)
        predicted = predicted[remaining_idx]
        y = y[remaining_idx]
        samples += y.shape[0]
        correct += (predicted == y).sum().item()

    return {"accuracy" : correct/samples}



def eval_op_ensemble_tr_lf_attack(models, loader):
    for model in models: 
        model.train()
    samples, correct = 0, 0
    trigger_value = 1
    source= 0
    target = 2
    with torch.no_grad():
      for i, (x, y) in enumerate(loader):
        remaining_idx = list(range(x.shape[0]))
        x, y = x.to(device), y.to(device)
        for example_id in range(x.shape[0]):
            if y[example_id] ==source:
                y[example_id]= target 
            else:
                remaining_idx.remove(example_id)
        y_ = torch.mean(torch.stack([model(x) for model in models], dim=0), dim=0)
        _, predicted = torch.max(y_.detach(), 1)
        predicted = predicted[remaining_idx]
        y = y[remaining_idx]
        samples += y.shape[0]
        correct += (predicted == y).sum().item()

    return {"accuracy" : correct/samples}


def reduce_average(target, sources):
    # import pdb; pdb.set_trace()
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].detach() for source in sources]), dim=0).clone()


def reduce_median(target, sources):
    for name in target:
        #   import pdb; pdb.set_trace()
        target[name].data = torch.median(torch.stack([source[name].detach() for source in sources]),
                                         dim=0).values.clone()
    #   import pdb; pdb.set_trace()


def reduce_trimmed_mean(target, sources, mali_ratio):
    import math
    trimmed_mean_beta = math.ceil(mali_ratio * len(sources)) + 1
    for name in target:
        stacked_weights = torch.stack([source[name].detach() for source in sources])
        #   import pdb; pdb.set_trace()
        user_num = stacked_weights.size(0)
        largest_value, _ = torch.topk(stacked_weights, k=trimmed_mean_beta, dim=0)
        smallest_value, _ = torch.topk(stacked_weights, k=trimmed_mean_beta, dim=0, largest=False)
        target[name].data = ((
                                     torch.sum(stacked_weights, dim=0)
                                     - torch.sum(largest_value, dim=0)
                                     - torch.sum(smallest_value, dim=0)
                             ) / (user_num - 2 * trimmed_mean_beta)).clone()
    #   import pdb; pdb.set_trace()


def reduce_krum(target, sources, mali_ratio):
    import math
    krum_mal_num = math.ceil(mali_ratio * len(sources)) + 1
    user_num = len(sources)
    user_flatten_grad = []
    for source in sources:
        user_flatten_grad_i = []
        for name in target:
            user_flatten_grad_i.append(torch.flatten(source[name].detach()))
        user_flatten_grad_i = torch.cat(user_flatten_grad_i)
        user_flatten_grad.append(user_flatten_grad_i)
    user_flatten_grad = torch.stack(user_flatten_grad)

    # compute l2 distance between users
    user_scores = torch.zeros((user_num, user_num), device=user_flatten_grad.device)
    for u_i, source in enumerate(sources):
        user_scores[u_i] = torch.norm(
            user_flatten_grad - user_flatten_grad[u_i],
            dim=list(range(len(user_flatten_grad.shape)))[1:],
        )
        # import pdb; pdb.set_trace()
        user_scores[u_i, u_i] = float('inf')
        topk_user_scores, _ = torch.topk(
            user_scores, k=user_num - krum_mal_num - 2, dim=1, largest=False
        )
    sm_user_scores = torch.sum(topk_user_scores, dim=1)

    # users with smallest score is selected as update gradient
    u_score, select_ui = torch.topk(sm_user_scores, k=1, largest=False)
    select_ui = select_ui.cpu().numpy()
    select_ui = select_ui[0]
    # print(select_ui)
    # import pdb; pdb.set_trace()
    for name in target:
        target[name].data = sources[select_ui][name].detach().clone()


def reduce_residual(source_1, source_2):
    tmp_dict = {}
    # import pdb; pdb.set_trace()
    for name in source_1:
        tmp_dict[name] = (source_1[name].detach() - source_2[name].detach()).clone()
        # import pdb; pdb.set_trace()
    return tmp_dict


def reduce_weighted(target, sources, weights):
    for name in target:
        # import pdb; pdb.set_trace()
        target[name].data = torch.sum(weights * torch.stack([source[name].detach() for source in sources], dim=-1),
                                      dim=-1).clone()
        # import pdb; pdb.set_trace()


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def copy(target, source):
    for name in target:
        target[name].data = source[name].detach().clone()


def olr(mu, var):
    from scipy.stats import multivariate_normal
    X = np.linspace(0, 0.4, 1000)
    if mu[0] > mu[1]:
        new_mu = [mu[1], mu[0]]
        new_var = [var[1], var[0]]
    else:
        new_mu = mu
        new_var = var
    step = 500
    x_step = (new_mu[1] - new_mu[0]) / step

    G_m = multivariate_normal(mean=new_mu[0], cov=new_var[0])
    G_b = multivariate_normal(mean=new_mu[1], cov=new_var[1])

    y_benign = G_b.pdf(X)
    y_mali = G_m.pdf(X)
    index = 0
    while index < step:
        x = mu[0] + x_step * index
        if G_b.pdf(x) > G_m.pdf(x):
            break
        index += 1
    overlap = (1 - G_m.cdf(x)) + G_b.cdf(x)
    return overlap
