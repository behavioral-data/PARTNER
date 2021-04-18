import torch
import codecs
import numpy as np


import pandas as pd
import re
import csv
import numpy as np

import time

from sklearn.metrics import f1_score

from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

from transformers import AdamW, RobertaConfig, RobertaForSequenceClassification

import datetime


class CoherenceClassifier():

	def __init__(self, 
			device,
			model_path = '/projects/bdata/talklife/dssg/ashish/Codes/Rewriting/RL_agent/models/rewards/coherence3.pt',
			batch_size=2):

		self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
		self.batch_size = batch_size
		self.device = device

		self.model = RobertaForSequenceClassification.from_pretrained(
			"roberta-base", # Use the 12-layer BERT model, with an uncased vocab.
			num_labels = 2, # The number of output labels--2 for binary classification.
							# You can increase this for multi-class tasks.   
			output_attentions = False, # Whether the model returns attentions weights.
			output_hidden_states = False, # Whether the model returns all hidden-states.
		)

		self.model = torch.nn.DataParallel(self.model)

		weights = torch.load(model_path)
		self.model.load_state_dict(weights)

		self.model.to(self.device)


	def predict_empathy(self, original_responses, candidate):

		input_ids = []
		attention_masks = []

		for idx, elem in enumerate(original_responses):

			response_sentence = original_responses[idx] + ' </s> ' + candidate

			encoded_dict = self.tokenizer.encode_plus(
								response_sentence,                      # Sentence to encode.
								add_special_tokens = True, # Add '[CLS]' and '[SEP]'
								max_length = 64,           # Pad & truncate all sentences.
								pad_to_max_length = True,
								return_attention_mask = True,   # Construct attn. masks.
								return_tensors = 'pt',     # Return pytorch tensors.
						)
			
			input_ids.append(encoded_dict['input_ids'])
			attention_masks.append(encoded_dict['attention_mask'])

		input_ids = torch.cat(input_ids, dim=0)
		attention_masks = torch.cat(attention_masks, dim=0)

		dataset = TensorDataset(input_ids, attention_masks)

		dataloader = DataLoader(
			dataset, # The test samples.
			sampler = SequentialSampler(dataset), # Pull out batches sequentially.
			batch_size = self.batch_size # Evaluate with this batch size.
		)

		self.model.eval()

		for batch in dataloader:
			b_input_ids = batch[0].to(self.device)
			b_input_mask = batch[1].to(self.device)

			with torch.no_grad():
				(logits, ) = self.model(input_ids = b_input_ids, 
														token_type_ids=None,
														attention_mask=b_input_mask,)

			logits = logits.detach().cpu().numpy().tolist()
			predictions = np.argmax(logits, axis=1).flatten()

		return (logits, predictions)


'''
Example:
'''
device = torch.device("cuda")
original_responses = [ 'I am so sorry that she is not getting it.','so she can get a better idea of what the condition entails?']
#sentences = ['why do you feel this way?', 'Let me know if you want to talk.']
candidate = 'Have you thought of directing her to sites like NAMI and Mental Health First Aid'
candidate2 = ' I have been on and off medication for the majority of my life '

coherence_classifier = CoherenceClassifier(device)

(logits, predictions,) = coherence_classifier.predict_empathy(original_responses, candidate)

print(logits, predictions)

(logits, predictions,) = coherence_classifier.predict_empathy(original_responses, candidate2)
print(logits, predictions)