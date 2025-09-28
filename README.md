# FedRE

This is the official implementation of FedRE: Model-Heterogeneous Federated Learning via Representation Entanglement
 
## Requirements

To run this project, make sure you have the following Python packages installed:

- `ujson`
- `scikit-learn`
- `h5py`
- `click`
- `calmsize`
- `opacus`
- `cvxpy`

## Data Preparation

1. Navigate to the `dataset` directory:

   ```bash
   cd ./dataset
   ```

2. **Dirichlet Distribution:**

   To generate data with a Dirichlet distribution:

   ```bash
   python generate_Cifar10.py noniid - dir
   ```

3. **Pathological Distribution:**

   To generate data with a Pathological distribution:

   ```bash
   python generate_Cifar10.py noniid - pat
   ```

## Running the Experiments

For detailed parameter settings, please refer to Appendix C of the paper. The parameter configurations can be adjusted for different practical applications. Below are some specific examples.

### 1. Train FedRE on Cifar10

To train the FedRE model on the MNIST dataset with the following settings:
- Global rounds: 100
- Local learning rate: 0.06
- Batch size: 32
- Model heterogeneity setting

Run the following command:

```bash
python main.py -data Cifar10 -algo FedRE -gr 100 -lr 0.06 -lbs 32
```

### 2. Train FedRE on Cifar100

To train the FedRE model on the Cifar100 dataset with the following settings:
- Global rounds: 100
- Local learning rate: 0.06
- Batch size: 32
- Model heterogeneity setting

Run the following command:

```bash
python main.py -data Cifar100 -algo FedRE -gr 100 -lr 0.06 -lbs 32
```

### 3. Train FedRE on TinyImagenet

To train the FedRE model on the TinyImagenet dataset with the following settings:
- Global rounds: 100
- Local learning rate: 0.06
- Batch size: 64
- Model heterogeneity setting

Run the following command:

```bash
python main.py -data TinyImagenet -algo FedRE -gr 100 -lr 0.06 -lbs 64
```

### Contact

If you have any problem with our code or have some suggestions, including the future feature, feel free to contact

Jiaqi Wu (1134608851@qq.com)

Yuan Yao (yaoyuan.hitsz@gmail.com)

or describe it in Issues.

---
