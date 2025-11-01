# Python Partial and Complementary Label Learning Library

This project compares complementary label learning (CLL) and partial label learning (PLL). It provides a Python framework for experimenting with state-of-the-art algorithms on CIFAR datasets.

## Implemented Strategies

The library implements several algorithms for PLL and CLL:

| Strategies | Type | Description |
| --- | --- | --- |
| [PiCo](https://arxiv.org/abs/2201.08984) | PLL | A contrastive learning framework for robust representation learning. |
| [Proden](https://arxiv.org/pdf/1803.09364) | PLL | A progressive denoising method that rectifies noisy labels during training. |
| [SoLar](https://arxiv.org/abs/2209.10365) | PLL | A self-organizing label refinement strategy using Sinkhorn's algorithm for label disambiguation. |
| [MCL (LOG)](https://openreview.net/pdf?id=SJzR2iRcK7) | CLL | Multi-class learning with Logarithmic Loss (LOG). |
| [MCL (MAE)](https://openreview.net/pdf?id=SJzR2iRcK7) | CLL | Multi-class learning with Mean Absolute Error (MAE) Loss. |
| [MCL (EXP)](https://openreview.net/pdf?id=SJzR2iRcK7) | CLL | Multi-class learning with Exponential (EXP) Loss. |

## Dataset Generation

The script generates PLL and CLL datasets from **CIFAR-10** and **CIFAR-20**. It also supports the human-annotated **CLCIFAR-10** and **CLCIFAR-20** datasets.

For generated datasets, the following parameters are available:

| Parameter | Options | Description |
| --- | --- | --- |
| `Type` | `constant`, `variable` | Specifies if the number of labels per sample is fixed. |
| `Value` | | - If `constant`, `Value` (k) is the number of partial labels. <br> - If `variable`, `Value` (q) is the probability of a false label being included. |
| `Noise` | `noisy`, `clean` | Introduces label noise if set to 'noisy'. |
| `eta` | | Sets the noise level for noisy data. |

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/esoau/pypcl.git](https://github.com/esoau/pypcl.git)
    cd pypcl
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run experiments using `run_experiment.py`. Hyperparameters can be set via command-line arguments or in `config.yaml`.

| Parameter | Description |
| --- | --- |
| `--dataset` | Dataset to use (`cifar10`, `cifar20`, `clcifar10`, `clcifar20`). |
| `--batch_size` | Batch size for training. |
| `--epochs` | Number of training epochs. |
| `--lr` | Learning rate. |
| `--weight_decay` | Weight decay for the optimizer. |
| `--momentum` | Momentum for the optimizer. |

**Example 1:**

Train models with a constant of 2 partial labels and no noise:

```bash
python scripts/run_experiment.py --dataset cifar10 --type constant --value 2 --noise clean
```

**Example 2:**

Example 2: CIFAR-20 with Variable Labels and Noise Run all algorithms on CIFAR-20. First, 20% label noise is applied to the ground truth. Then, partial labels are generated where each false label has a 50% probability of being included in the candidate set.

```bash
python scripts/run_experiment.py --dataset cifar20 --type variable --value 0.5 --noise noisy --eta 0.2
```

**Example 3:**

CLCIFAR-10 (Human-Annotated) Run all algorithms on the pre-defined CLCIFAR-10 dataset.

```bash
python scripts/run_experiment.py --dataset clcifar10
```
