# Towards Practical Non-Adversarial Distribution Matching

---
## Overview

This repository contains the code and data for the paper titled "Towards Practical Non-Adversarial Distribution Matching". The paper explores distribution alignment using VAE-based methods.


## Paper Abstract

Distribution matching can be used to learn invariant representations with applications in fairness and robustness.
Most prior works resort to adversarial matching methods 
but the resulting minimax problems are unstable and challenging to optimize.
Non-adversarial likelihood-based approaches either require model invertibility, impose constraints on the latent prior, or lack a generic framework for distribution matching.
To overcome these limitations, we propose a non-adversarial VAE-based matching method that can be applied to any model pipeline.
We develop a set of alignment upper bounds \changeziyu{for distribution matching} (including a noisy bound) that have VAE-like objectives but with a different perspective.
We carefully compare our method to prior VAE-based matching approaches both theoretically and empirically.
Finally, we demonstrate that our novel matching losses can replace adversarial losses in standard invariant representation learning pipelines without modifying the original architectures---thereby significantly broadening the applicability of non-adversarial matching methods.

---

## Environment Setup

`
conda env create -f environment.yml
`

## Dataset

1. Adult Dataset. We adopted the preprocessed Adult Dataset from [Conditional Learning of Fair Representations](https://github.com/hanzhaoml/ICLR2020-CFair).
2. [USPS](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps).
3. [MNIST](http://yann.lecun.com/exdb/mnist/).

## Code Structure

- `backbone/`: Contains the backbone models that are used in MNIST_USPS experiment.
- `config/`: Contains the hyperparameters that are used in MNIST_USPS experiment.
- `dataset/`: Contains the loading function for different dataset in the experiment section.
- `models/`: Contains the models used in the experiment section.
- `notebooks/`: Contains the notebooks which gives interactive notebooks for the experiment.
- `train_script.py` Contains some demo scripts for running MNIST_USPS experiment.

---

## Citation

If you find this work useful in your research, please consider citing:

```
@misc{gong2023practical,
      title={Towards Practical Non-Adversarial Distribution Alignment via Variational Bounds}, 
      author={Ziyu Gong and Ben Usman and Han Zhao and David I. Inouye},
      year={2023},
      eprint={2310.19690},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

---

## Contact
Please email to [gong123@purdue.edu](mailto:gong123@purdue.edu) if you have any questions or comments.