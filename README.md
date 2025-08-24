# Python Partial and Complementary Label Learning Library

This project, developed at the NTU Computation Learning Lab, focuses on the comparison of complementary label learning (CLL) with multiple labels and partial label learning (PLL). It provides a Python-based framework for experimenting with various state-of-the-art algorithms in these domains. The experiments are conducted on the **CIFAR-10** dataset, with functionalities to generate both PLL and CLL datasets with different characteristics.

## Implemented Strategies

The library implements and compares several state-of-the-art algorithms for both PLL and CLL:

| Strategies                                       | Type          | Description                                                                                                                                  |
| ------------------------------------------------ | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [PiCo](https://arxiv.org/abs/2201.08984)           | PLL           | A contrastive learning framework that uses a momentum-based queue of features to learn robust representations.                                |
| [Proden](https://arxiv.org/pdf/1803.09364)         | PLL           | A progressive denoising method that rectifies noisy labels during the training process, leading to improved model performance.                |
| [SoLar](https://arxiv.org/abs/2209.10365)          | PLL           | A self-organizing label refinement strategy that employs Sinkhorn's algorithm for efficient and accurate label disambiguation.              |
| [MCL](https://openreview.net/pdf?id=SJzR2iRcK7)    | CLL           | A multi-class learning approach with three loss variations: **MAE** (Mean Absolute Error), **LOG** (Logarithmic Loss), and **EXP** (Exponential Loss). |

## Dataset Generation

The script allows for the generation of various PLL and CLL datasets from CIFAR-10, with the ability to control the following parameters:

| Parameter | Options              | Description                                                                                                                                    |
| --------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `Type`    | `constant`, `variable` | Specifies whether the number of labels per sample is fixed or not.                                                                           |
| `Value`   |                      | - If `Type` is 'constant', `Value` (k) specifies the number of partial labels. <br> - If `Type` is 'variable', `Value` (q) is the probability of a false label being included in the candidate set. |
| `Noise`   | `noisy`, `clean`     | Use 'noisy' to introduce label noise or 'clean' for a dataset without it.                                                                    |
| `eta`     |                      | When using noisy data, this sets the noise level.                                                                                              |

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/esoau/pypcl.git](https://github.com/esoau/pypcl.git)
    cd pypcl
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run an experiment, use the `run_experiment.py` script with the desired arguments for dataset generation and model training. You can adjust hyperparameters for data generation, training, and the specific algorithms (PiCo and SoLar) through the use of arguments whhen running the program or by modifying the config.yaml file. The arguments are as follows:

| Parameter        | Description                |
| ---------------- | -------------------------- |
| `--batch_size`   | Batch size for training.   |
| `--epochs`       | Number of training epochs. |
| `--lr`           | Learning rate.             |
| `--weight_decay` | Weight decay for the optimizer. |
| `--momentum`     | Momentum for the optimizer. |

**Example:**

To train the models on a dataset with a constant number of 2 partial labels and no noise, run the following command:

```bash
python scripts/run_experiment.py --type constant --value 2 --noise clean
```
