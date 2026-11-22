# CS434 Assignment 3: Graph Neural Networks for Traffic Forecasting

This repository contains the code, results, and report for Assignment 3. Questions 1 and 2 study message passing on small road-sensor examples. Questions 3 to 7 forecast one hour of traffic flow for the 716 San Diego sensors of the LargeST dataset, using a temporal-only baseline and spatio-temporal graph neural networks with fixed, adaptive, and hybrid adjacency matrices.

Repository: https://github.com/syeddaniyalg/cs434-pa3

## Repository Structure

```
cs434-pa3/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py            paths, dates, and default hyperparameters
│   ├── data.py              raw file reading, resampling, splits, scaling, batching
│   ├── graph.py             adjacency normalization and graph statistics
│   ├── adaptive.py          fixed, adaptive, and hybrid adjacency modules
│   ├── models.py            temporal encoder, graph convolution, forecast head
│   ├── train.py             training loop, evaluation, checkpointing
│   ├── metrics.py           masked MAE, RMSE, and MAPE
│   └── utils.py             seeding, logging, figure saving, hardware info
├── scripts/
│   ├── q1_verify.py         Question 1 checks
│   ├── q2_ring.py           Question 2 ring-road experiments
│   ├── download_data.py     LargeST download
│   ├── prepare_data.py      Question 3(a)
│   ├── explore_graph.py     Questions 3(b) and 3(c)
│   ├── run_experiment.py    Questions 4 and 5, one model per run
│   └── analyze.py           Question 6 tables, figures, and graph analysis
├── data/                    created by the scripts, not tracked
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── figures/             all generated figures (PNG and PDF)
│   ├── tables/              comparison tables (CSV, Markdown, LaTeX)
│   ├── results/             metrics, histories, logs, learned adjacencies
│   └── checkpoints/         model checkpoints, not tracked
└── paper/
    ├── main.tex             report source
    ├── neurips_2026.sty     NeurIPS style file
    └── figures/             figures selected for the report
```

## Environment

The experiments were run with Python 3.11, PyTorch 2.5.1, and CUDA 12.4. A GPU is optional; the code falls back to the CPU automatically.

```
conda create -n torch_env python=3.11
conda activate torch_env
pip install -r requirements.txt
```

Install the CUDA build of PyTorch that matches your system if a GPU is used. PyTables is not required, because the raw HDF5 files are read with `h5py`.

## Data

The project uses the San Diego subset of LargeST: the 716 sensors in Caltrans District 11, from 2020 and 2021. The download script fetches four files from the Kaggle dataset `liuxu77/largest` into `data/raw/`:

| File | Content |
|---|---|
| `ca_meta.csv` | sensor metadata for California |
| `ca_rn_adj.npy` | road-distance adjacency for California |
| `ca_his_raw_2020.h5` | 5-minute flow for 2020 |
| `ca_his_raw_2021.h5` | 5-minute flow for 2021 |

Kaggle may require a signed-in account. The files can also be downloaded manually and placed in `data/raw/` with the names above. Kaggle delivers some files as zip archives under their original names; `prepare_data.py` detects this and extracts them in place. About 15 GB of free disk space is needed after extraction.

## Reproducing the Results

All commands are run from the repository root. Each script writes its results to `outputs/` and logs its progress.

**Questions 1 and 2.** These scripts need no dataset.

```
python -m scripts.q1_verify
python -m scripts.q2_ring
```

**Question 3.** Download the data, resample it, and compute the graph statistics.

```
python -m scripts.download_data
python -m scripts.prepare_data
python -m scripts.explore_graph
```

**Questions 4 and 5.** Train the four models. Each run takes about 3 to 5 minutes on an RTX 4050 Laptop GPU.

```
python -m scripts.run_experiment --model temporal --fresh
python -m scripts.run_experiment --model fixed --fresh
python -m scripts.run_experiment --model adaptive --fresh
python -m scripts.run_experiment --model hybrid --fresh
```

**Question 6.** Build the comparison tables, prediction plots, and learned-graph analysis.

```
python -m scripts.analyze
```

**Report.** The report compiles with any LaTeX distribution, for example:

```
cd paper
tectonic main.tex
```

## Training Options

`run_experiment.py` accepts every default in `src/config.py` as a command-line flag.

| Flag | Default | Meaning |
|---|---|---|
| `--model` | required | `temporal`, `fixed`, `adaptive`, or `hybrid` |
| `--epochs` | 50 | maximum number of epochs |
| `--patience` | 10 | early stopping patience on validation MAE |
| `--batch_size` | 64 | training batch size |
| `--lr` | 0.001 | Adam learning rate |
| `--weight_decay` | 0.0001 | weight decay, not applied to the learned graph parameters |
| `--hidden` | 32 | hidden feature dimension |
| `--embed_dim` | 10 | node embedding dimension for the learned graph |
| `--graph_layers` | 1 | number of graph convolution layers |
| `--topk` | 0 | keep only the top-k learned neighbors (0 keeps all) |
| `--alpha` | learn | hybrid mixing weight, `learn` or a fixed value such as `0.5` |
| `--ckpt_every` | 2 | checkpoint interval in epochs |
| `--fresh` | off | delete previous checkpoints and start again |

Training resumes automatically from `outputs/checkpoints/<run>/last.pt` if a run is interrupted. A checkpoint is also saved whenever validation MAE improves, and evaluation always uses the checkpoint with the lowest validation MAE. The random seed is fixed at 42.

## Method Summary

| Component | Choice |
|---|---|
| Resampling | mean of three 5-minute readings, as in the LargeST benchmark |
| Missing values | forward fill; filled values are masked out of the loss and all metrics |
| Splits | train 2020, validation January to March 2021, test April to December 2021 |
| Windows | 8 input steps (2 hours), 4 output steps (1 hour), built inside each split |
| Normalization | z-score from observed training values only |
| Temporal encoder | two 1D convolutions (kernel 3, the second with dilation 2) and a linear layer |
| Graph layer | H = Z + ReLU(A Z W), with A replaced by the identity for the temporal-only model |
| Forecast head | two-layer MLP predicting four steps for every sensor |
| Fixed adjacency | row normalization D^-1 (A + I), because the road graph is directed |
| Adaptive adjacency | Softmax(ReLU(E1 E2^T)); sensors never observed in training are excluded as sources |
| Hybrid adjacency | alpha A_fixed + (1 - alpha) A_adaptive, with alpha learned |
| Loss | masked MAE in original flow units |

## Results

Test results in original flow units (vehicles per 5 minutes).

| Model | Adjacency | MAE | RMSE | MAPE (%) | Training time (s) |
|---|---|---|---|---|---|
| Temporal-only | None | 24.20 | 35.68 | 18.64 | 206.0 |
| STGNN | Fixed | 23.17 | 34.33 | 17.14 | 249.2 |
| STGNN | Adaptive | 20.91 | 31.67 | 13.71 | 234.4 |
| STGNN | Hybrid | 21.73 | 32.77 | 13.98 | 173.5 |

The full analysis, including per-horizon errors and the comparison between learned and road connections, is in the report.

## Hardware

| Item | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB |
| CPU | Intel Core, 14 cores and 20 threads |
| Memory | 15.7 GB |
| Software | Windows 11, Python 3.11, PyTorch 2.5.1, CUDA 12.4 |

## References

1. Xu Liu et al. LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting. NeurIPS Datasets and Benchmarks Track, 2023. https://github.com/liuxu77/LargeST
2. Thomas N. Kipf and Max Welling. Semi-Supervised Classification with Graph Convolutional Networks. ICLR, 2017.
3. Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, and Chengqi Zhang. Graph WaveNet for Deep Spatial-Temporal Graph Modeling. IJCAI, 2019.
