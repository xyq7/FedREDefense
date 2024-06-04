import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import tqdm
from utils import kd_loss, DiffAugment
import copy
from torch.utils.data import Dataset
from copy import deepcopy
from torchvision.utils import save_image
import time
import random
from reparam_module import ReparamModule
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
mean_dataset={
    "cifar10":  [0.4914, 0.4822, 0.4465],
    "mnist":  [0.1307],
    "fmnist":  [0.1307],
}
std_dataset  = {
    "cifar10" : [0.2023, 0.1994, 0.2010],
    "mnist" : [0.3081],
    "fmnist" : [0.3081],
}

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

def reduce_params(sources, weights):
    targets = []
    for i in range(len(sources[0])):
        target = torch.sum(weights * torch.stack([source[i].cuda() for source in sources], dim = -1), dim=-1)
        targets.append(target)
    return targets

class Synthesizer:
    def __init__(self, network, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset =args.dataset
        self.batch_syn = args.batch_syn
        self.save_path = args.RESULTS_PATH
        self.iteration = args.Iteration
        self.Max_Iter = args.Max_Iter
        self.channel = args.channel
        hard_label = [np.ones(args.ipc, dtype=np.long)*i for i in range(args.num_classes)]
        label_syn = torch.nn.functional.one_hot(torch.tensor(hard_label).reshape(-1), num_classes=args.num_classes).float()
        label_syn = label_syn * args.label_init
        label_syn = label_syn.detach().to(self.device).requires_grad_(True)
        image_syn = torch.randn(size=(args.num_classes * args.ipc, args.channel, args.imsize[0], args.imsize[1]), dtype=torch.float)
        syn_lr = torch.tensor(args.lr_teacher).to(self.device)
        image_syn = image_syn.detach().to(self.device).requires_grad_(True)
        syn_lr = syn_lr.detach().to(self.device).requires_grad_(True)
        if args.img_optim == "sgd":
            optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
            optimizer_label = torch.optim.SGD([label_syn], lr=args.lr_label, momentum=0.5)
        else:
            optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_img)
            optimizer_label = torch.optim.Adam([label_syn], lr=args.lr_label)
        if args.lr_optim == "sgd":
            optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
        else:
            optimizer_lr = torch.optim.Adam([syn_lr], lr=args.lr_lr)
        self.ori_image, self.ori_label, self.ori_syn_lr = copy.deepcopy(image_syn), copy.deepcopy(label_syn), copy.deepcopy(syn_lr)
        
        self.label_syn, self.image_syn, self.syn_lr = label_syn, image_syn, syn_lr
        self.optimizer_img, self.optimizer_label, self.optimizer_lr = optimizer_img, optimizer_label, optimizer_lr
        self.network = network.cuda()
        self.syn_steps= args.syn_steps


    def reinitilize_date(self, args):
        self.label_syn, self.image_syn, self.syn_lr =copy.deepcopy(self.ori_label),  copy.deepcopy(self.ori_image), copy.deepcopy(self.ori_syn_lr)
        if args.img_optim == "sgd":
            self.optimizer_img = torch.optim.SGD([self.image_syn], lr=args.lr_img, momentum=0.5)
            self.optimizer_label = torch.optim.SGD([self.label_syn], lr=args.lr_label, momentum=0.5)
        else:
            self.optimizer_img = torch.optim.Adam([self.image_syn], lr=args.lr_img)
            self.optimizer_label = torch.optim.Adam([self.label_syn], lr=args.lr_label)
        if args.lr_optim == "sgd":
            self.optimizer_lr = torch.optim.SGD([self.syn_lr], lr=args.lr_lr, momentum=0.5)
        else:
            self.optimizer_lr = torch.optim.Adam([self.syn_lr], lr=args.lr_lr)

    def follow_data(self, args,syn_data, syn_label, lr_syn,id):
        self.label_syn, self.image_syn, self.syn_lr =copy.deepcopy(syn_label[id]),  copy.deepcopy(syn_data[id]), copy.deepcopy(lr_syn[id])
        if args.img_optim == "sgd":
            self.optimizer_img = torch.optim.SGD([self.image_syn], lr=args.lr_img, momentum=0.5)
            self.optimizer_label = torch.optim.SGD([self.label_syn], lr=args.lr_label, momentum=0.5)
        else:
            self.optimizer_img = torch.optim.Adam([self.image_syn], lr=args.lr_img)
            self.optimizer_label = torch.optim.Adam([self.label_syn], lr=args.lr_label)
        if args.lr_optim == "sgd":
            self.optimizer_lr = torch.optim.SGD([self.syn_lr], lr=args.lr_lr, momentum=0.5)
        else:
            self.optimizer_lr = torch.optim.Adam([self.syn_lr], lr=args.lr_lr)
        
    def synthesize_single(self, start_trajectories,end_trajectories,syn_data, syn_label, lr_syn,  id, args, c_round):

        if len(syn_data[id]) == 0:
            self.reinitilize_date(args)
            iters = self.iteration
        else:
            assert len(syn_label[id]) != 0
            self.follow_data(args,syn_data, syn_label, lr_syn,id)
            iters = self.Max_Iter
        
        true_iter = -1
        for it in range(0, iters):
            start_trajectory = start_trajectories[id][-1]
            end_trajectory = end_trajectories[id][-1]
            
            student_net = ReparamModule(copy.deepcopy(self.network))
            student_net.train()
            num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])
            starting_params = start_trajectory
            target_params = end_trajectory

            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)
            student_params = [torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
            starting_params = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0)
            
            syn_images = self.image_syn
            y_hat = self.label_syn
            param_loss_list = []
            param_dist_list = []
            indices_chunks = []
            for step in range(self.syn_steps):
                if not indices_chunks:
                    indices = torch.randperm(len(syn_images))
                    indices_chunks = list(torch.split(indices, self.batch_syn))

                these_indices = indices_chunks.pop()
                x = syn_images[these_indices]
                this_y = y_hat[these_indices]
                forward_params = student_params[-1]
                x = student_net(x, flat_param=forward_params)
                ce_loss = kd_loss(x, this_y)
                
                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
                
                student_params.append(student_params[-1] - self.syn_lr * grad)


            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)

            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum") + 1e-9
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum") + 1e-9

            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)


            param_loss /= num_params
            param_dist /= num_params

            param_loss /= param_dist

            grand_loss = param_loss
            if grand_loss.detach().cpu() < 0.6:
                true_iter = it + 1
                break
            self.optimizer_img.zero_grad()
            self.optimizer_label.zero_grad()
            self.optimizer_lr.zero_grad()

            grand_loss.backward()

            self.optimizer_img.step()
            self.optimizer_lr.step()
            self.optimizer_label.step()

            

            for _ in student_params:
                del _
        if true_iter != -1:
            iters = true_iter
        return grand_loss.item(), self.image_syn,self.label_syn, self.syn_lr, iters
