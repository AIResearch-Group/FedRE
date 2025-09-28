import copy
import time
import torch
import torch.nn as nn
import numpy
from flcore.clients.clientre import clientre
from flcore.servers.serverbase import Server
from threading import Thread
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
from flcore.clients.clientbase import load_item, save_item
import numpy as np
import os

def custom_collate_fn(batch):
    entangle_reps, weight_label_tuples = zip(*batch)
    return torch.stack(entangle_reps, dim=0), weight_label_tuples

class FedRE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.global_model = None
        self.set_slow_clients()
        self.set_clients(clientre)
        self.protos = None
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate
        self.head_batch_size=args.head_batch_size
        head = load_item(self.clients[0].role, 'model', self.clients[0].save_folder_name).head
        save_item(head, 'Server', 'head', self.save_folder_name)
        self.algorithm = args.algorithm
        self.global_rounds = args.global_rounds

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
            for client in self.selected_clients:
                    client.train()
            
            self.receive_entangle_rep()
            self.train_head()
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

    def receive_entangle_rep(self):
        assert (len(self.selected_clients) > 0)
        self.uploaded_entangle_rep = []
        for client in self.selected_clients:
            client_batchentangle_rep = client.entangle_rep()
            self.uploaded_entangle_rep.extend(client_batchentangle_rep)

    def train_head(self):
        head = load_item('Server', 'head', self.save_folder_name)
        opt_h = torch.optim.SGD(head.parameters(), lr=self.server_learning_rate)
        entangle_rep_loader = DataLoader(self.uploaded_entangle_rep, self.head_batch_size, drop_last=False, shuffle=True, collate_fn=custom_collate_fn)
        for epoch in range(1):
            total_loss = 0

            for entangle_rep, weight_label_tuples in entangle_rep_loader:
                entangle_rep = entangle_rep.to(self.device)
                pred = head(entangle_rep)
                loss = 0.0
                for i in range(len(pred)):
                    total_sample_loss = 0.0
                    for weight, label in weight_label_tuples[i]:
                        label = torch.tensor([label], dtype=torch.long, device=self.device)
                        single_loss = weight * self.CEloss(pred[i].unsqueeze(0), label)
                        total_sample_loss += single_loss
                    loss += total_sample_loss
                opt_h.zero_grad()
                loss.backward()
                opt_h.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(entangle_rep_loader)}")
        save_item(head, 'Server', 'head', self.save_folder_name)





