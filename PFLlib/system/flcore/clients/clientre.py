import random
import copy
import torch
import numpy as np
import torch.nn as nn
import time
from flcore.clients.clientbase import Client
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

class clientre(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.global_protos = None
        self.scheduler_local = StepLR(self.optimizer, step_size=50, gamma=0.5)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
  
    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def set_head_parameters(self, head):
        for new_param, old_param in zip(head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()
            
    def set_adapter_parameters(self, adapter):
        for new_param, old_param in zip(adapter.parameters(), self.model.adapter.parameters()):
            old_param.data = new_param.data.clone()

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()
        reps = []
        all_batches_protos = []
        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)
        self.protos = agg_func(protos)
        for cc in self.protos.keys():
            y = torch.tensor(cc, dtype=torch.int64, device=self.device)
            all_batches_protos.append((self.protos[cc], y))
        return all_batches_protos

    def entangle_rep(self):
        entangle_rep_tuple = []
        protos = self.collect_protos()
        weights = torch.rand(len(protos)).to(self.device)
        weights = weights / weights.sum()
        entangle_rep = torch.zeros_like(protos[0][0])
        for (proto, _), weight in zip(protos, weights):
            entangle_rep += weight * proto
        weight_label_tuples = [(weight.item(), label) for (proto, label), weight in zip(protos, weights)]
        entangle_rep_tuple.append((entangle_rep, weight_label_tuples))
        return entangle_rep_tuple
    
def agg_func(protos):

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos