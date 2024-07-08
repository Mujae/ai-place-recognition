# AI-Place-Recognition

For all main contributors, please check [contributing](#contributing).

## Introduction

This repository is dedicated to research and development of AI-based place recognition.

## How To Use

### Clone 

Clone this GitHub repository:

```
git clone https://github.com/SKHU-Adventure/ai-place-recognition.git
cd ai-place-recognition
```

### Requirements

The main branch works with **CUDA 12.1**, **CUDNN 8.9.2**, **NCCL 2.18.3** with **Python 3.8**.
Refer to a [document](docs/environment.md) for pre-setting and then install requirements:

```bash
pip install -r requirements.txt
```

### Prepare Datasets

1. Prepare dataset for training: 

For public datasets, you may refer to following links.
- [Nordland](https://drive.google.com/drive/folders/1CzzLo-t9iLYOszcHAnB3KaWwkP5jyJn1?usp=sharing)
- [Tokyo](https://www.di.ens.fr/willow/research/netvlad/) (available on request)

### How to Train

1. Create a directory for your experiment (e.g., `experiments/sample`)

2. Write a setup.ini file in the directory (e.g., `experiments/sample/setup.ini`).

2. Run:
```bash
python3 train.py [EXPERIMENT_DIR]
```
where `[EXPERIMENT_DIR]` is the directory created above.

### How to Evaluate


### How to Demo


## Contributing

Main contributors:

- [Mujae Park](https://github.com/Mujae), ``mujae9837[at]gmail.com``
- [Younguk Jeon](https://github.com/jayiuk), ``jayiuk987[at]gmail.com``
- [Heeju Cha](https://github.com/JOOZOO20), ``qwsa7896[at]naver.com``
- [Younah Kim](https://github.com/kkiwiio), ``kkiwiio[at]gmail.com``

Advisior:
- [Sangyun Lee](https://sylee-skhu.github.io), ``sylee[at]skhu.ac.kr``
