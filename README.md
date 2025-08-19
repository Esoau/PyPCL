# Python Partial and Complementary Label Learning Library
NTU Computation Learning Lab project on comparison of complementary label learning with multiple labels and partial label learning
This is a Python script designed to experiment with complementary and partial label learning. It generated PLL and CLL datasets from **CIFAR10**. It implements various CLL and PLL methods listed below:
| Strategies                                                 | Type             | Description                                                  |
| ---------------------------------------------------------- | ---------------- | ------------------------------------------------------------ |
| [PiCo](https://arxiv.org/pdf/1705.07541)                   | None             |A contrastive learning framework that uses a momentum-based queue of features.|
| [Proden](https://arxiv.org/pdf/1705.07541)                 | None             |A progressive denoising method that rectifies labels during training.|
| [MCL](https://arxiv.org/pdf/1705.07541)                    | MAE, LOG, EXP    |A multi-class learning approach with MAE, LOG, and EXP loss variations.|
| [SoLar](https://arxiv.org/pdf/1705.07541)                  | None             |A self-organizing label refinement strategy using Sinkhorn's algorithm.|

Additionally, the models can be trained on variants of partial and complementary label datasets generated from the datasets. 
| Parameter                                                  | Options            | Description                                                  |
| ---------------------------------------------------------- | ------------------ | ------------------------------------------------------------ |
| Type                                                       | constant, variable | Whether the number of labels is constant or variable         |
| Value                                                      |                    | q value for variable, k value for constant                   |
| Noise                                                      | noisy, clean       | Whether there are false labels(noise) in the dataset         |
