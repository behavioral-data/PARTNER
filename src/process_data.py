#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
"""
preprocess input data into feature and stores binary as python shelve DB
each chunk is gzipped JSON string
"""
import argparse
import gzip
import json
import subprocess as sp
import shelve
import os
from os.path import dirname, exists, join

import torch
from lsp_model_rl import GPT2Tokenizer
from tqdm import tqdm

from env import END_OF_TEXT_TOKEN
from gpt2_training.train_utils import InputFeatures as InputFeatures


def _get_file_len(corpus):
	n_line = int(sp.check_output(f"wc -l {corpus}".split(),
								 universal_newlines=True).split()[0])
	return n_line


def _norm_text(text):
	w, *toks = text.strip().split()
	try:
		w = float(w)
	except Exception:
		toks = [w] + toks
		w = 1.0
	return w, ' '.join(toks)


def _get_inputs_from_text(text, tokenizer):
	src_seeker_post = text.strip().split('\t')[0].strip() # srcs.split('<SPLIT>')[0].strip()[4:]
	src_response_post = text.strip().split('\t')[1].strip() # srcs.split('<SPLIT>')[1].strip()

	srcs = src_seeker_post + ' <SPLIT> ' + src_response_post + ' <END>'

	inputs = []

	for src in srcs.split(' EOS '):
		context_id = tokenizer.encode(src)
		inputs.append(context_id)
	
	# pos = [pos,]
	return inputs, src_seeker_post, src_response_post


def _make_features(id_, inputs, src_seeker_post, src_response_post, tokenizer, max_len):
	end_of_text_id = tokenizer.encoder[END_OF_TEXT_TOKEN]
	features = []
	sents = []
	ws = []
	ps = []
	len_ = 0
	i = 0
	for ids in inputs:
		if len(ids) > max_len:
			ids = ids[:max_len]

		len_ += (len(ids) + 1)
		sents.append(ids)

	if len(sents) >= 1:		
		feat = _make_feature(id_ + i, sents, src_seeker_post, src_response_post, end_of_text_id)
		if feat is not None:
			features.append(feat)

	return features


def _make_feature(id_, sents, src_seeker_post, src_response_post, eos):
	input_ids = [i for s in sents for i in s+[eos]][:-1]

	weights = []
	token_type_ids = []  # this becomes round ids

	split_id = toker.encode("<SPLIT>")[0]

	curr_id = 0

	for i, s in enumerate(input_ids):

		if s == split_id:
			curr_id = 1

		token_type_ids.append(curr_id)


	# TODO: handle trailing -1's
	# input_ids = input_ids[:i+1]
	# token_type_ids = token_type_ids[:i+1]

	# pad to multiples of 8
	while len(input_ids) % 8 != 0:
		input_ids.append(0)
		token_type_ids.append(0)

	position_ids = list(range(len(input_ids)))
	assert (len(input_ids) == len(position_ids) == len(token_type_ids))
	assert len(input_ids) % 8 == 0

	if len(input_ids) == 0:
		import pdb
		pdb.set_trace()

	feature = InputFeatures(id_, input_ids, position_ids, token_type_ids,
						 src_seeker_post, src_response_post)

	return feature


toker = GPT2Tokenizer.from_pretrained('gpt2-medium')
toker.add_tokens(['<SPLIT>', '<START>', '<END>'])

def main(args):

	attrs = []
	if args.reverse:
		attrs.append('reverse')
	if args.two_turn:
		attrs.append('2turn')
	if attrs:
		db_path = (f'{args.corpus[:-4]}.{args.max_seq_len}len.'
				   f'{".".join(attrs)}.db/db')
	else:
		db_path = f'{args.corpus[:-4]}.{args.max_seq_len}len.db/db'
	if exists(dirname(db_path)):
		raise ValueError('Found existing DB, please backup')
	else:
		os.makedirs(dirname(db_path))
	with open(args.corpus, "r", encoding="utf-8") as reader, \
			shelve.open(db_path, 'n') as db:
		chunk = []
		n_chunk = 0
		n_example = 0
		for line in tqdm(reader, total=_get_file_len(args.corpus)):
			# print('line:', line)
			# print('n_chunk:', len(chunk))

			try:
				if len(chunk) >= args.chunk_size:
					# save and renew chunk
					db[f'chunk_{n_chunk}'] = gzip.compress(
						json.dumps(chunk[:args.chunk_size]).encode('utf-8'))
					chunk = chunk[args.chunk_size:]
					n_chunk += 1

				inputs, src_seeker_post, src_response_post  = _get_inputs_from_text(line, toker)
				if args.reverse:
					inputs = list(reversed(inputs))
				if args.two_turn:
					inputs = inputs[:2]

				features = _make_features(n_example, inputs, src_seeker_post, src_response_post,
										  toker, args.max_seq_len)

				# print('features:', features)
				for feature in features:
					chunk.append(vars(feature))
					n_example += 1
			except Exception as e:
				print('!!! prepro exception !!!', e)
				continue
		# save last chunk
		db[f'chunk_{n_chunk}'] = gzip.compress(
			json.dumps(chunk).encode('utf-8'))
	# save relevant information to reproduce
	meta = {'n_example': n_example,
			'chunk_size': args.chunk_size,
			'max_seq_len': args.max_seq_len,
			'reverse': args.reverse,
			'two_turn': args.two_turn}
	with open(join(dirname(db_path), 'meta.json'), 'w') as writer:
		json.dump(meta, writer, indent=4)
	torch.save(toker, join(dirname(db_path), 'tokenizer.pt'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--corpus', required=True,
						help='file name of training corpus (should be .tsv)')
	parser.add_argument('--chunk_size', type=int, default=65536,
						help='num of data examples in a storing chunk')
	parser.add_argument('--max_seq_len', type=int, default=128,
						help='discard data longer than this')
	parser.add_argument('--reverse', action='store_true',
						help='reverse the src tgt')
	parser.add_argument('--two_turn', action='store_true',
						help='take only the first 2 turns')

	args = parser.parse_args()

	main(args)
