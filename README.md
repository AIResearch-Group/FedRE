# FedRE

This is the official implementation of **FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning**, accepted to CVPR 2026.

Paper is now available online: https://arxiv.org/pdf/2511.22265
 
## Requirements

Create and activate the environment from the repository root:

```bash
conda env create -f env_cuda_latest.yaml
conda activate FedRE
```

Use matching PyTorch and torchvision builds for your CUDA version. CPU execution is available with `-dev cpu`.

The original environment recipe above has not been rebuilt in this revision. GPU smoke tests used Python 3.12.3, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, NumPy 1.26.4, SciPy 1.16.3, scikit-learn 1.9.1, and h5py 3.16.0.

## Implementations

- `HtFLlib`: model-heterogeneous FedRE with `HtM10` (CNN, GoogLeNet, MobileNetV2, five ResNets, and two ViTs).
- `PFLlib`: model-homogeneous FedRE with `-m cnn` or `-m resnet18`.

All HtM10 models map their representations to `feature_dim` (512 by default). Ordinary models use a linear classifier; ViTs retain two linear layers with Tanh and use num_classes as the intermediate width. Input sizes are 32x32 for CIFAR-10/100 and 64x64 for TinyImageNet.

The server retains one linear classifier, `feature_dim -> num_classes`. ViT heads are initialized as `feature_dim -> num_classes -> num_classes` with Tanh between the layers. This makes the second layer accept the first layer's output after the original FedGH parameter broadcast. The existing broadcast copies the server classifier into the ViT's first linear layer; the second layer remains local and is updated during local training. It does not synchronize the entire two-layer ViT classifier. No server-head grouping is used.

## Data Preparation

Generate data inside the implementation you intend to run. For heterogeneous experiments, start from the repository root:

```bash
cd HtFLlib/dataset
python generate_cifar10.py noniid - dir
python generate_cifar100.py noniid - dir
python generate_TinyImagenet.py noniid - dir
```

For homogeneous experiments, use `PFLlib/dataset` instead. Replace the final `dir` argument with `pat` for the pathological partition:

```bash
python generate_cifar10.py noniid - pat
```

Both partition commands write into the same dataset directory, so keep separate copies if you need both partitions. The scripts use 10 clients and a 3:1 local train/test split. Dirichlet alpha is 0.1. Pathological partitions assign 2, 10, or 20 classes per client for CIFAR-10, CIFAR-100, or TinyImageNet, respectively.

HtFLlib's [dataset instructions](https://github.com/TsingZ0/HtFLlib#scenarios-and-datasets) refer to PFLlib for dataset preparation. TinyImageNet follows that [upstream generator](https://github.com/TsingZ0/PFLlib/blob/master/dataset/generate_TinyImagenet.py): use the official **training images only** (100,000 images, 200 classes, 500 images per class), then create client train/test splits. The official validation directory is not merged into this pool. FedRE's existing client count and random seed are retained.

**Existing TinyImageNet shards must be regenerated after this fix.** Older generators incorrectly treated all validation images as class 0. Move the old `TinyImagenet` directory aside before regenerating; raw images can be reused in the new `TinyImagenet/rawdata` directory. A matching old `config.json` causes the generator to skip generation, so rerunning on the old directory does not repair its labels.

## Model-Heterogeneous Experiments

After preparing data in `HtFLlib/dataset`, enter `HtFLlib/system`:

```bash
cd ../system
python main.py -data Cifar10 -nb 10 -algo FedRE -m HtM10 -gr 100 -lr 0.06 -lbs 32 -hbs 10 -slr 0.01
```

CNN flatten dimensions and ViT image sizes are selected from the dataset name; no source edits are needed when switching between these datasets. Keep `-nb` consistent with the dataset.

The command above retains the paper's local learning rate. Earlier miniature tests with `-lr 0.06` produced nonfinite losses, including the original-broadcast CIFAR-10 baseline. Bounded correctness tests use an explicit `-lr 0.001`; they do not validate convergence or paper accuracy. With the class-dependent ViT width, all ten models completed two-round GPU smoke tests on CIFAR-10, CIFAR-100, and TinyImageNet, using 640 training and 160 held-out images per dataset, with finite training losses and evaluation values.

## Model-Homogeneous Experiments

Prepare data in `PFLlib/dataset`, then enter `PFLlib/system`:

```bash
cd ../system
python main.py -data Cifar10 -nb 10 -algo FedRE -m cnn -gr 100 -lr 0.007 -lbs 32 -sbs 10 -slr 0.01
python main.py -data Cifar100 -nb 100 -algo FedRE -m cnn -gr 100 -lr 0.01 -lbs 32 -sbs 10 -slr 0.01
python main.py -data TinyImagenet -nb 200 -algo FedRE -m resnet18 -gr 100 -lr 0.01 -lbs 64 -sbs 10 -slr 0.01
```

These settings follow the paper's detailed experimental setup. For homogeneous CIFAR-10 with the pathological partition, use local learning rate `0.008` instead of `0.007`. The server learning rate is controlled by `-slr`; the head batch-size flags are `-hbs` in HtFLlib and `-sbs` in PFLlib.

PFLlib supports optional exponential learning-rate decay, for example `-ld True -ldg 0.99`; use `-ld False` for a fixed rate. HtFLlib uses a fixed local learning rate. Boolean options accept `True`/`False` and `1`/`0`.

The existing round numbering and evaluation schedule are retained. Framework options for early stopping, client dropout, and time-based filtering are not implemented by this FedRE training loop.

## Citation

If you find this work helpful, please consider citing:
```
@inproceedings{yao2026fedre,
  title     = {FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning},
  author    = {Yao, Yuan and Wang, Lixu and Wu, Jiaqi and Song, Jin and Chen, Simin and Wang, Zehua and Tian, Zijian and Chen, Wei and Li, Huixia and Li, Xiaoxiao},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {39466--39475},
  year      = {2026}
}
```

## Contact

If you have any problem with our code or have some suggestions, including the future feature, feel free to contact

Jiaqi Wu (1134608851@qq.com)

Yuan Yao (yaoyuan.hitsz@gmail.com)

or describe it in Issues.

---
