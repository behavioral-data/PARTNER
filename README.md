# PARTNER: Empathic Rewriting using Reinforcement Learning

If this code or dataset helps you in your research, please cite the following publication:
```bash
@inproceedings{sharma2021facilitating,
    title={Towards Facilitating Empathic Conversations in Online Mental Health Support: A Reinforcement Learning Approach},
    author={Sharma, Ashish and Lin, Inna W and Miner, Adam S and Atkins, David C and Althoff, Tim},
    year={2021},
    booktitle={WWW}
}
```

## Introduction

We work towards improving empathy in online mental health support conversations. We introduce a new task of empathic rewriting which aims to transform low-empathy conversational posts to higher empathy. Learning such transformations is challenging and requires a deep understanding of empathy
while maintaining conversation quality through text fluency and specificity to the conversational context. Here we propose Partner, a deep reinforcement learning (RL) agent that learns to make sentence-level edits to posts in order to increase the expressed level of empathy while maintaining conversation quality. Our RL agent leverages a policy network, based on a transformer language model adapted from GPT-2, which performs the dual task of generating candidate empathic sentences and adding those sentences at appropriate positions. During training, we reward transformations that increase empathy in posts while maintaining text fluency, context specificity, and diversity.

## Quickstart

### 1. Prerequisites

Our framework can be compiled on Python 3 environments. The modules used in our code can be installed using:
```
$ pip install -r requirements.txt
```


### 2. Prepare dataset

To be added soon!

### 3. Training the model
For training our model on the sample input data, run the following command:

```
python train.py \
	--model_name_or_path models/medium/ \
	--train_input_file data.db \
	--output_dir output/ \
	--log_dir output/ \
	--train_batch_size 4 \
	--num_optim_steps 500000 \
	--MI
```