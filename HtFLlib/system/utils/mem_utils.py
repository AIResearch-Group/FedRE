import math
import gc
from collections import defaultdict
from typing import Optional, Tuple, List

import torch
  
from math import isnan
from calmsize import size as calmsize

def readable_size(num_bytes: int) -> str:
    return '' if isnan(num_bytes) else '{:.2f}'.format(calmsize(num_bytes))

LEN = 79
PYTORCH_MIN_ALLOCATE = 2 ** 9
PYTORCH_MIN_CACHE = 2 ** 20

class MemReporter():

    def __init__(self, model: Optional[torch.nn.Module] = None):
        self.tensor_name = {}
        self.device_mapping = defaultdict(list)
        self.device_tensor_stat = {}
        self.name_idx = 0

        tensor_names = defaultdict(list)
        if model is not None:
            assert isinstance(model, torch.nn.Module)

            for name, param in model.named_parameters():
                tensor_names[param].append(name)

        for param, name in tensor_names.items():
            self.tensor_name[id(param)] = '+'.join(name)

    def _get_tensor_name(self, tensor: torch.Tensor) -> str:
        tensor_id = id(tensor)
        if tensor_id in self.tensor_name:
            name = self.tensor_name[tensor_id]
        else:
            name = type(tensor).__name__ + str(self.name_idx)
            self.tensor_name[tensor_id] = name
            self.name_idx += 1
        return name

    def collect_tensor(self):

        objects = gc.get_objects()
        tensors = [obj for obj in objects if isinstance(obj, torch.Tensor)]
        for t in tensors:
            self.device_mapping[t.device].append(t)

    def get_stats(self):

        visited_data = {}
        self.device_tensor_stat.clear()

        def get_tensor_stat(tensor: torch.Tensor) -> List[Tuple[str, int, int, int]]:

            assert isinstance(tensor, torch.Tensor)

            name = self._get_tensor_name(tensor)
            if tensor.is_sparse:
                indices_stat = get_tensor_stat(tensor._indices())
                values_stat = get_tensor_stat(tensor._values())
                return indices_stat + values_stat

            numel = tensor.numel()
            element_size = tensor.element_size()
            fact_numel = tensor.storage().size()
            fact_memory_size = fact_numel * element_size

            memory_size = math.ceil(fact_memory_size / PYTORCH_MIN_ALLOCATE) \
                    * PYTORCH_MIN_ALLOCATE

            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                name = '{}(->{})'.format(
                    name,
                    visited_data[data_ptr],
                )

                memory_size = 0
            else:
                visited_data[data_ptr] = name

            size = tuple(tensor.size())
            if not size:
                size = (1,)

            return [(name, size, numel, memory_size)]

        for device, tensors in self.device_mapping.items():
            tensor_stats = []
            for tensor in tensors:

                if tensor.numel() == 0:
                    continue
                stat = get_tensor_stat(tensor) 
                tensor_stats += stat
                if isinstance(tensor, torch.nn.Parameter):
                    if tensor.grad is not None:
                        self.tensor_name[id(tensor.grad)] = '{}.grad'.format(
                            self._get_tensor_name(tensor)
                        )
                        stat = get_tensor_stat(tensor.grad)
                        tensor_stats += stat

            self.device_tensor_stat[device] = tensor_stats

        self.device_mapping.clear()

    def print_stats(self, verbose: bool = False, target_device: Optional[torch.device] = None) -> None:
        for device, tensor_stats in self.device_tensor_stat.items():
            if target_device is not None and device != target_device:
                continue

            print('\nStorage on {}'.format(device))
            total_mem = 0
            total_numel = 0
            for stat in tensor_stats:
                name, size, numel, mem = stat

                total_mem += mem
                total_numel += numel

            print('-'*LEN)
            print('Total Tensors: {} \tUsed Memory: {}'.format(
                total_numel, readable_size(total_mem),
            ))

            if device != torch.device('cpu'):
                with torch.cuda.device(device):
                    memory_allocated = torch.cuda.memory_allocated()
                print('The allocated memory on {}: {}'.format(
                    device, readable_size(memory_allocated),
                ))
                if memory_allocated != total_mem:
                    print('Memory differs due to the matrix alignment or'
                          ' invisible gradient buffer tensors')
            print('-'*LEN)

    def report(self, verbose: bool = False, device: Optional[torch.device] = None) -> None:

        self.collect_tensor()
        self.get_stats()
        self.print_stats(verbose, target_device=device)