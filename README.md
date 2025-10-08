# Python Partial and Complementary Label Learning Library

This project compares complementary label learning (CLL) and partial label learning (PLL). It provides a Python framework for experimenting with state-of-the-art algorithms on CIFAR datasets.

## Implemented Strategies

The library implements several algorithms for PLL and CLL:

| Strategies                                       | Type          | Description                                                                                                                                  |
| ------------------------------------------------ | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [PiCo](https://arxiv.org/abs/2201.08984)           | PLL           | A contrastive learning framework for robust representation learning.                                                                         |
| [Proden](https://arxiv.org/pdf/1803.09364)         | PLL           | A progressive denoising method that rectifies noisy labels during training.                                                                  |
| [SoLar](https://arxiv.org/abs/2209.10365)          | PLL           | A self-organizing label refinement strategy using Sinkhorn's algorithm for label disambiguation.                                             |
| [MCL (MAE/LOG/EXP)](https://openreview.net/pdf?id=SJzR2iRcK7)    | CLL           | A multi-class learning approach with three loss variations: Mean Absolute Error (MAE), Logarithmic Loss (LOG), and Exponential Loss (EXP). |

## Dataset Generation

The script generates PLL and CLL datasets from CIFAR-10 with controllable parameters:

| Parameter | Options              | Description                                                                                                                                    |
| --------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `Type`    | `constant`, `variable` | Specifies if the number of labels per sample is fixed.                                                                                         |
| `Value`   |                      | - If `constant`, `Value` (k) is the number of partial labels. <br> - If `variable`, `Value` (q) is the probability of a false label being included. |
| `Noise`   | `noisy`, `clean`     | Introduces label noise if set to 'noisy'.                                                                                                      |
| `eta`     |                      | Sets the noise level for noisy data.                                                                                                           |

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/esoau/pypcl.git
    cd pypcl
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run experiments using `run_experiment.py`. Hyperparameters can be set via command-line arguments or in `config.yaml`.

| Parameter        | Description                     |
| ---------------- | ------------------------------- |
| `--batch_size`   | Batch size for training.        |
| `--epochs`       | Number of training epochs.      |
| `--lr`           | Learning rate.                  |
| `--weight_decay` | Weight decay for the optimizer. |
| `--momentum`     | Momentum for the optimizer.     |

**Example:**

Train models with a constant of 2 partial labels and no noise:

```bash
python scripts/run_experiment.py --type constant --value 2 --noise clean
```
