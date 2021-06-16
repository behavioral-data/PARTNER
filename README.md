# PARTNER: Empathic Rewriting using Reinforcement Learning
This repository contains code for our [WWW 2021 publication](https://arxiv.org/pdf/2101.07714.pdf) on empathic rewriting. The repository is based on [DialoGPT](https://github.com/microsoft/DialoGPT) and uses a similar code structure and environment.

If this code helps you in your research, please cite:
```bash
@inproceedings{sharma2021facilitating,
    title={Towards Facilitating Empathic Conversations in Online Mental Health Support: A Reinforcement Learning Approach},
    author={Sharma, Ashish and Lin, Inna W and Miner, Adam S and Atkins, David C and Althoff, Tim},
    year={2021},
    booktitle={TheWebConf}
}
```

## Introduction

We work towards improving empathy in online mental health support conversations. We introduce a new task of empathic rewriting which aims to transform low-empathy conversational posts to higher empathy. Learning such transformations is challenging and requires a deep understanding of empathy while maintaining conversation quality through text fluency and specificity to the conversational context. Here we propose PARTNER, a deep reinforcement learning (RL) agent that learns to make sentence-level edits to posts in order to increase the expressed level of empathy while maintaining conversation quality.

For a quick overview, check out [bdata.uw.edu/empathy](http://bdata.uw.edu/empathy/). For a detailed description of our work, please read our [WWW 2021 publication](https://arxiv.org/pdf/2101.07714.pdf).


## Quickstart

### 1. Setup and Installation

Our framework can be compiled on Python 3 environments. The modules used in our code can be installed using:
```
$ pip install -r requirements.txt
```


### 2. Prepare dataset

A sample raw input data file is available in [dataset/sample_data.tsv](dataset/sample_data.tsv). Each line in the file has a post and a corresponding response (tab-separated). This input file can be converted into a format that is recognized by the model using with following command:
```
python src/process_data.py --corpus dataset/sample_data.tsv
```

Running this command will generate a folder named `sample_data.128len.db`.

### 3. Training the model
For training our model on the sample input data, run the following command:

```
python src/train_model.py \
	--model_name_or_path models/medium/ \
	--train_input_file dataset/sample_data.128len.db \
	--output_dir output/ \
	--log_dir output/ \
	--train_batch_size 4 \
	--num_optim_steps 500
```

Note that before running this code, you will need a DialoGPT-like transformer model for initialization (`model_name_or_path`, ideally finetuned on your dataset, check the warm-start strategy in the paper) and will need to separately train multiple reward functions. For training the empathy reward, check out [this repository](https://github.com/behavioral-data/Empathy-Mental-Health). 
