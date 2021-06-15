# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

from __future__ import absolute_import, division, print_function, unicode_literals


import logging
import copy
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Iterable, Optional, Tuple
from torch import Tensor
import time
import nltk
import numpy as np

from transformers import GPT2PreTrainedModel, GPT2Model

from pytorch_pretrained_bert.modeling_gpt2 import GPT2LMHead, Attention, Block, \
	LayerNorm, MLP

# from generate import top_filtering

from .rewards import calc_rewards

# from .generation_utils import GenerationMixin


logger = logging.getLogger(__name__)


class AttentionFP16(Attention):
	def __init__(self, nx, n_ctx, config, scale=False):
		super(AttentionFP16, self).__init__(nx, n_ctx, config, scale)

	def _attn(self, q, k, v):
		w = torch.matmul(q, k)
		if self.scale:
			w = w / math.sqrt(v.size(-1))
		nd, ns = w.size(-2), w.size(-1)
		b = self.bias[:, :, ns-nd:ns, :ns]
		w = w * b - 1e4 * (1 - b)    # point out by Yen-Chun, FP16 overflow

		w = nn.Softmax(dim=-1)(w)
		return torch.matmul(w, v)


class BlockFP16(Block):
	def __init__(self, n_ctx, config, scale=False):
		super(BlockFP16, self).__init__(n_ctx, config, scale)
		nx = config.n_embd
		self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
		self.attn = AttentionFP16(nx, n_ctx, config, scale)
		self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
		self.mlp = MLP(4 * nx, config)


class GPT2ModelFP16(GPT2Model):
	def __init__(self, config):
		# super(GPT2ModelFP16, self).__init__(config)
		super().__init__(config)
		self.wte = nn.Embedding(config.vocab_size, config.n_embd)
		self.wpe = nn.Embedding(config.n_positions, config.n_embd)
		block = BlockFP16(config.n_ctx, config, scale=True)
		self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
		self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

		self.init_weights()

class GPT2LMHeadModel(GPT2PreTrainedModel):
	def __init__(self, config):
		super(GPT2LMHeadModel, self).__init__(config)
		self.transformer = GPT2Model(config)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # GPT2LMHead(self.transformer.wte.weight, config)
		self.position_num_labels = 2
		self.lambda_position = 0.1
		self.position_classifier = GPT2ClassificationHead(num_labels = self.position_num_labels) #GPT2LMHead(self.transformer.wte.weight, config)
		self.init_weights()

	def set_tied(self):
		""" Make sure we are sharing the embeddings
		"""
		self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

	def get_output_embeddings(self):
		return self.lm_head
	
	def padding_tensor_3D(self, sequences, max_len):
		"""
		:param sequences: list of tensors
		:return:
		"""
		num = len(sequences)
		out_dims = (num, max_len, *sequences[0].shape[1:])
		out_tensor = sequences[0].data.new(*out_dims).fill_(0)

		# print('out_tensor:', out_tensor.shape)

		mask = sequences[0].data.new(*out_dims).fill_(0)
		for i, tensor in enumerate(sequences):
			length = tensor.size(0)
			# print('length:', length)
			out_tensor[i, :length] = tensor
			mask[i, :length] = 1
		return out_tensor, mask

	def padding_tensor_2D(self, sequences, max_len):
		"""
		:param sequences: list of tensors
		:return:
		"""
		num = len(sequences)
		out_dims = (num, max_len)
		out_tensor = sequences[0].data.new(*out_dims).fill_(0)

		# print('out_tensor:', out_tensor.shape)

		mask = sequences[0].data.new(*out_dims).fill_(0)
		for i, tensor in enumerate(sequences):
			length = min(tensor.size(0), max_len)
			# print('length:', length)
			out_tensor[i, :length] = tensor[:length]
			mask[i, :length] = 1
		return out_tensor, mask

	def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, position_labels=None, past=None, seeker_post=None, response_post=None, top_k=60, top_p=0.92, temperature=0.9, eos=None, tokenizer=None, baseline_val=0):

		transformer_start_time = time.time()

		# Forward Transformer Pass
		hidden_states, presents = self.transformer(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, past=past)

		transformer_end_time = time.time()

		# Get LM and position logits
		lm_logits = self.lm_head(hidden_states)

		if tokenizer is None:
			return lm_logits, presents

		position_logits = self.position_classifier(hidden_states[:, -1, :])

		# A1 (Selecting a position)
		probs_position = torch.softmax(position_logits.view(-1, self.position_num_labels), -1) # (batch_size, num_position)
		all_positions = torch.argmax(probs_position, 1)
		all_positions = all_positions.squeeze()

		all_positions = all_positions.cpu().numpy().tolist()

		# A2 (Candidate Sentence): Sample from DialoGPT
		all_outputs = []
		all_output_ids = []
		all_output_logits = []
		sample_dialogpt_start_time = time.time()

		for ii, _ in enumerate(input_ids):
			curr_seeker = tokenizer.encode(seeker_post[ii] + tokenizer.eos_token)
			curr_seeker = torch.tensor([curr_seeker,])
			curr_seeker = curr_seeker.to('cuda')
			generated_output = self.generate(input_ids = curr_seeker, max_length=1000, pad_token_id=tokenizer.eos_token_id, top_p=0.92, top_k=60, temperature=1, num_return_sequences=1)

			curr_output = tokenizer.decode(generated_output[:, curr_seeker.shape[-1]:][0], skip_special_tokens=True)

			curr_output_ids = generated_output[:, curr_seeker.shape[-1]:][0]
			curr_output_ids = curr_output_ids[:hidden_states.shape[1]]
			curr_position_ids = torch.tensor(range(len(curr_output_ids)), dtype=torch.long).to("cuda")

			curr_output_logits = lm_logits[ii, range(curr_output_ids.shape[0]), curr_output_ids]

			all_outputs.append(curr_output)
			all_output_ids.append(curr_output_ids)
			all_output_logits.append(curr_output_logits)

		log_softmax = nn.LogSoftmax(1)

		all_output_logits, _ = self.padding_tensor_2D(all_output_logits, hidden_states.shape[1])
		all_output_logits = log_softmax(all_output_logits)

		sample_dialogpt_end_time = time.time()


		# Calculate Reward 
		
		rewritten_response = []

		for idx, _ in enumerate(all_outputs):
			curr_seeker_post = seeker_post[idx]
			curr_response = response_post[idx]
			curr_output = all_outputs[idx]
			curr_position = all_positions[idx]

			curr_response_li = nltk.sent_tokenize(curr_response)

			if curr_position == 0:
				curr_rewritten_response = curr_response

			else:
				curr_rewritten_response_li = curr_response_li[:curr_position] + [curr_output] + curr_response_li[curr_position:]
				curr_rewritten_response = '. '.join(curr_rewritten_response_li)
			
			rewritten_response.append(curr_rewritten_response)

		reward_start_time = time.time()
		reward = calc_rewards(seeker_post, response_post, rewritten_response, _empathy_change=True, _perplexity=True)
		reward_end_time = time.time()

		batches = np.arange(input_ids.shape[0]).tolist()

		rl_loss = - (reward - baseline_val) * (-torch.mean(all_output_logits[batches,]) + torch.mean(torch.log(probs_position[batches, all_positions]) ))

		return rl_loss, reward

	
	def forward_pointwise(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
		hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
		# import pdb; pdb.set_trace()
		lm_logits = self.lm_head(hidden_states)
		if lm_labels is not None:
			# loss_fct = CrossEntropyLoss(ignore_index=-1)
			# loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
			loss_fct1 = CrossEntropyLoss(ignore_index=-1, reduction='none')
			loss1 = loss_fct1(lm_logits.view(-1, lm_logits.size(-1)),
							  lm_labels.view(-1))
			loss1 = loss1.view(lm_labels.size(0), lm_labels.size(1))
			label_size = torch.sum(lm_labels != -1, dim=1).type(loss1.type())
			loss1 = torch.sum(loss1, dim=1)/label_size
			ppl1 = torch.exp(loss1)

			return loss1, ppl1
		return lm_logits, presents
	
	def prepare_inputs_for_generation(self, input_ids, **kwargs):
		return {"input_ids": input_ids}

class GPT2ClassificationHead(nn.Module):
	"""Head for sentence-level classification tasks."""

	def __init__(self, hidden_dropout_prob=0.1, hidden_size=1024, num_labels=2):
		super().__init__()

		self.dense = nn.Linear(hidden_size, hidden_size)
		self.dropout = nn.Dropout(hidden_dropout_prob)
		self.out_proj = nn.Linear(hidden_size, num_labels)

	def forward(self, features, **kwargs):
		x = features[:, :]
		x = self.dropout(x)
		x = self.dense(x)
		x = torch.relu(x)
		x = self.dropout(x)
		x = self.out_proj(x)
		return x
