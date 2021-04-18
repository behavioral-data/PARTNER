from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams
import numpy as np
from numpy import dot
from numpy.linalg import norm

#from bert_serving.client import BertClient
#bc = BertClient()

import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

device = 'cuda'
GPT_model = GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium').to(device)
GPT_tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')

GPT_tokenizer.pad_token = GPT_tokenizer.eos_token

GPT_model.eval()

MAX_LEN = 64

# BLEU
def bleu(predicted, target):

	all_bleus = []

	smoothie = SmoothingFunction().method4

	for idx, elem in enumerate(predicted):
		try:
			curr_bleu = sentence_bleu([str(target[idx])], str(predicted[idx]), smoothing_function=smoothie)
			all_bleus.append(curr_bleu)
		except ZeroDivisionError:
			continue

	return np.mean(all_bleus)


# Perplexity
def perplexity(predicted):

	BATCH_SIZE = 1

	tokenized_input = GPT_tokenizer.batch_encode_plus(predicted, max_length=MAX_LEN, pad_to_max_length=True, truncation=True)
	
	input_ids = tokenized_input['input_ids'] 
	attention_masks = tokenized_input['attention_mask']

	input_ids = torch.tensor(input_ids)
	attention_masks = torch.tensor(attention_masks)

	data = TensorDataset(input_ids, attention_masks)

	sampler = SequentialSampler(data)
	dataloader = DataLoader(data, sampler=sampler, batch_size = BATCH_SIZE)

	all_loss = []

	with torch.no_grad():

		for batch in dataloader:
			b_input = batch[0].to(device)
			b_attn = batch[1].to(device)

			outputs = GPT_model(b_input, attention_mask=b_attn, labels=b_input)
	
			loss, logits = outputs[:2]
			all_loss.append(loss.item())

	return math.exp(np.mean(all_loss))


# Diversity

# Distinct-1, Distinct-2 (# of unigrams and bigrams divided by the total number of words)

def distinct(predicted):
	
	UNIGRAMS = set()
	BIGRAMS = set()

	NUM_WORDS = 0

	for idx, elem in enumerate(predicted):
		curr_unigrams = ngrams(str(elem).split(), 1)
		curr_bigrams = ngrams(str(elem).split(), 2)

		NUM_WORDS += len(str(elem).split())

		for unigram in curr_unigrams:
			UNIGRAMS.add(' '.join(unigram).strip())
		
		for bigram in curr_bigrams:
			BIGRAMS.add(' '.join(bigram).strip())
		

	
	DISTINCT_1 = len(UNIGRAMS) / NUM_WORDS
	DISTINCT_2 = len(BIGRAMS) / NUM_WORDS

	return DISTINCT_1, DISTINCT_2


# Entropy
# def entropy():


# Specificity
def specificity(seeker_post, predicted):

	# Get embeddings
	seeker_post_embeddings = bc.encode(seeker_post)
	predicted_response_embeddings = bc.encode(predicted)
	
	# Compute cosine similarity

	all_cos_sim = []

	for idx, elem in enumerate(seeker_post_embeddings):
		a = seeker_post_embeddings[idx]
		b = predicted_response_embeddings[idx]

		cos_sim = dot(a, b)/(norm(a)*norm(b))
		all_cos_sim.append(cos_sim)
	
	return np.mean(all_cos_sim)