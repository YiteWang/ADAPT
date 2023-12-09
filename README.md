# Balanced Training for Sparse GANs [[Paper](https://arxiv.org/abs/2302.14670)]

Yite Wang*, Jing Wu*, Naira Hovakimyan, Ruoyu Sun

$*$ denotes equal contribution.

In NeurIPS'2023

### Requirements:

The code is tested using Redhat system with python 3.9. NVIDIA V100 and NVIDIA RTX 2080TI are used to run all the experiments. To install required packages, please find the `requirements.txt` file.

### Prepare dataset

1. CIFAR-10 and STL-10 datasets will download automatically.

2. Modify folder location of IS computation `MODEL_DIR` under `sparselearning/gan_utils/inception_score.py`.

3. Download FID statistics from this repo of [GNGAN](https://github.com/basiclab/GNGAN-PyTorch).

### Run our code:

Please see the scripts in `scripts` folder to run our code. For more information, please refer to `main.py` and `sparselearning/core.py`.

For example, to run the baseline: 

```
chmod +x scripts/baseline1.sh
scripts/baseline1.sh
```

### Acknowlegement:

Our code is mainly based on :

[ITOP](https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization) and [GAN ticket](https://github.com/VITA-Group/GAN-LTH).

### Contact:

Yite Wang (yitew2@illinois.edu)

Jing Wu (jingwu6@illinois.edu)