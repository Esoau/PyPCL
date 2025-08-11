# PLL CLL Comparitive Test
NTU Computation Learning Lab project on comparison of complementary label learning with multiple labels and partial label learning
This is a Python script designed to experiment with complementary and partial label learning. It generated PLL and CLL datasets from **CIFAR10**. It implements various CLL and PLL methods listed below:
| Strategies                                                 | Type             | Description                                                  |
| ---------------------------------------------------------- | ---------------- | ------------------------------------------------------------ |
| [PiCo](https://arxiv.org/pdf/1705.07541)                   | None             |                                                              |
| [Proden](https://arxiv.org/pdf/1705.07541)                 | None             |                                                              |
| [MCL](https://arxiv.org/pdf/1705.07541)                    | MAE, LOG, EXP    |                                                              |
| [SoLar](https://arxiv.org/pdf/1705.07541)                  | None             |                                                              |

Additionally, the models can be trained on variants of partial and complementary label datasets generated from the datasets. 
| Parameter                                                  | Options            | Description                                                  |
| ---------------------------------------------------------- | ------------------ | ------------------------------------------------------------ |
| Type                                                       | constant, variable | Whether the number of labels is constant or variable         |
| Value                                                      |                    | q value for variable, k value for constant                   |
| Noise                                                      | noisy, clean       | Whether there are false labels(noise) in the dataset         |
