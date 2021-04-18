from .roberta import RobertaForTokenClassification, RobertaModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import math
import torch.nn.functional as F

from .modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from .configuration_roberta import RobertaConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable

from transformers import GPT2Model
from transformers import AutoModelWithLMHead, AutoTokenizer


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
	"roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
	"roberta-talklike": "../../Empathy-Models/Pretraining-Tasks/pretrained-talklife-roberta/checkpoint-199000/pytorch_model.bin",
	"roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
	"roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
	"distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
	"roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
	"roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class Norm(nn.Module):
	def __init__(self, d_model, eps = 1e-6):
		super().__init__()
	
		self.size = d_model
		# create two learnable parameters to calibrate normalisation
		self.alpha = nn.Parameter(torch.ones(self.size))
		self.bias = nn.Parameter(torch.zeros(self.size))
		self.eps = eps
	def forward(self, x):
		norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
		/ (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
		return norm


class MultiHeadAttention(nn.Module):
	def __init__(self, heads, d_model, dropout = 0.1):
		super().__init__()
		
		self.d_model = d_model
		self.d_k = d_model // heads
		self.h = heads
		
		self.q_linear = nn.Linear(d_model, d_model)
		self.v_linear = nn.Linear(d_model, d_model)
		self.k_linear = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)
		self.out = nn.Linear(d_model, d_model)
	
	def forward(self, q, k, v, mask=None):
		
		bs = q.size(0)
		
		# perform linear operation and split into h heads
		
		k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
		q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
		v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
		
		# transpose to get dimensions bs * h * sl * d_model
	   
		k = k.transpose(1,2)
		q = q.transpose(1,2)
		v = v.transpose(1,2)
		# calculate attention using function we will define next
		scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
		
		# concatenate heads and put through final linear layer
		concat = scores.transpose(1,2).contiguous()\
		.view(bs, -1, self.d_model)
		
		output = self.out(concat)
	
		return output
	
	def attention(self, q, k, v, d_k, mask=None, dropout=None):
	
		scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
		
		if mask is not None:
			mask = mask.unsqueeze(1)
			scores = scores.masked_fill(mask == 0, -1e9)
		
		scores = F.softmax(scores, dim=-1)
			
		if dropout is not None:
			scores = dropout(scores)
			
		output = torch.matmul(scores, v)
		
		return output


class SeekerEncoder(BertPreTrainedModel):
	config_class = RobertaConfig
	pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
	base_model_prefix = "roberta"

	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels

		self.roberta = RobertaModel(config)



		# self.dropout = nn.Dropout(config.hidden_dropout_prob)
		# self.classifier = nn.Linear(config.hidden_size, config.num_labels)

		self.init_weights()
	
	def get_input_embeddings(self):
		return self.roberta.embeddings.word_embeddings

	def set_input_embeddings(self, value):
		self.roberta.embeddings.word_embeddings = value
	

class ResponderEncoder(BertPreTrainedModel):
	config_class = RobertaConfig
	pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
	base_model_prefix = "roberta"

	def __init__(self, config):
		super().__init__(config)
		# self.num_labels = config.num_labels

		self.roberta = RobertaModel(config)
		# self.dropout = nn.Dropout(config.hidden_dropout_prob)
		# self.classifier = nn.Linear(config.hidden_size, config.num_labels)

		self.init_weights()
	
	def get_input_embeddings(self):
		return self.roberta.embeddings.word_embeddings

	def set_input_embeddings(self, value):
		self.roberta.embeddings.word_embeddings = value



class EmpathyRationaleClassification(RobertaForTokenClassification):

	config_class = RobertaConfig
	pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
	base_model_prefix = "roberta"

	def __init__(self, config):
		super().__init__(config)

		self.roberta = RobertaModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.rationale_classifier = nn.Linear(config.hidden_size, config.rationale_num_labels)

		self.rationale_num_labels = config.rationale_num_labels
		self.empathy_num_labels = config.empathy_num_labels

		self.empathy_classifier = RobertaClassificationHead()

		self.init_weights()

	# @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		empathy_labels=None,
		rationale_labels=None,
		lambda_=0.1
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
			Labels for computing the token classification loss.
			Indices should be in ``[0, ..., config.num_labels - 1]``.
	Returns:
		:obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
		loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
			Classification loss.
		scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
			Classification scores (before SoftMax).
		hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
			Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
			of shape :obj:`(batch_size, sequence_length, hidden_size)`.
			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
			Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
			:obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
			heads.
	Examples::
		from transformers import RobertaTokenizer, RobertaForTokenClassification
		import torch
		tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
		model = RobertaForTokenClassification.from_pretrained('roberta-base')
		input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
		labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
		outputs = model(input_ids, labels=labels)
		loss, scores = outputs[:2]
		"""

		outputs = self.roberta(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)

		sequence_output = outputs[0]
		logits_empathy = self.empathy_classifier(sequence_output[:, 0, :])

		sequence_output = self.dropout(sequence_output)
		logits_rationales = self.rationale_classifier(sequence_output)

		outputs = (logits_empathy,logits_rationales) + outputs[2:]


		if rationale_labels is not None:
			loss_fct = CrossEntropyLoss()
			# Only keep active parts of the loss
			if attention_mask is not None:
				active_loss = attention_mask.view(-1) == 1
				active_logits = logits_rationales.view(-1, self.rationale_num_labels)
				active_labels = torch.where(
					active_loss, rationale_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(rationale_labels)
				)
				loss_rationales = loss_fct(active_logits, active_labels)
			else:
				loss_rationales = loss_fct(logits_rationales.view(-1, self.rationale_num_labels), rationale_labels.view(-1))
			
			# outputs = (loss_rationales,) + outputs


		if empathy_labels is not None:
			loss_fct = CrossEntropyLoss()
			loss_empathy = loss_fct(logits_empathy.view(-1, self.empathy_num_labels), empathy_labels.view(-1))

			# outputs = (loss_empathy,) + outputs

		loss = loss_empathy + lambda_ * loss_rationales

		outputs = (loss, loss_empathy, loss_rationales) + outputs

		return outputs  # (loss), (scores_empathy, scores_rationales), (hidden_states), (attentions)




class BiEncoderWithRationaleClassification(nn.Module):

	# config_class = RobertaConfig
	# pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
	# base_model_prefix = "roberta"

	def __init__(self, hidden_dropout_prob=0.1, rationale_num_labels=2, empathy_num_labels=3, hidden_size=768):
		super().__init__()

		# self.roberta_SP = RobertaModel(config)

		self.dropout = nn.Dropout(hidden_dropout_prob)
		
		self.rationale_classifier = nn.Linear(hidden_size, rationale_num_labels)

		self.ladder_hidden_SP_1 = nn.Linear(hidden_size, 268)
		self.ladder_hidden_SP_2 = nn.Linear(100, 100)
		
		self.ladder_hidden_RP_1 = nn.Linear(hidden_size, 500)
		self.ladder_hidden_RP_2 = nn.Linear(200, 200)

 
		# self.attn = nn.Linear(hidden_size, max_length)
		# self.attn_combine = nn.Linear(hidden_size, self.hidden_size)


		self.rationale_num_labels = rationale_num_labels
		self.empathy_num_labels = empathy_num_labels

		self.empathy_classifier = RobertaClassificationHead(hidden_size = 768)

		self.apply(self._init_weights)

		self.seeker_encoder = SeekerEncoder.from_pretrained(
								"/projects/bdata/talklife/dssg/ashish/Codes/Empathy-Models/Pretraining-Tasks/pretrained-talklife-roberta/checkpoint-199000/", # Use the 12-layer BERT model, with an uncased vocab.
								output_attentions = False, # Whether the model returns attentions weights.
								output_hidden_states = False)

		self.responder_encoder = ResponderEncoder.from_pretrained(
								"/projects/bdata/talklife/dssg/ashish/Codes/Empathy-Models/Pretraining-Tasks/pretrained-talklife-roberta/checkpoint-199000/", # Use the 12-layer BERT model, with an uncased vocab.
								output_attentions = False, # Whether the model returns attentions weights.
								output_hidden_states = False)

	
	def _init_weights(self, module):
		""" Initialize the weights """
		if isinstance(module, (nn.Linear, nn.Embedding)):
			# Slightly different from the TF version which uses truncated_normal for initialization
			# cf https://github.com/pytorch/pytorch/pull/5617
			initializer_range=0.02
			module.weight.data.normal_(mean=0.0, std=initializer_range)
		elif isinstance(module, BertLayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()


	# @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
	def forward(
		self,
		input_ids_SP=None,
		input_ids_RP=None,
		attention_mask_SP=None,
		attention_mask_RP=None,
		token_type_ids_SP=None,
		token_type_ids_RP=None,
		position_ids_SP=None,
		position_ids_RP=None,
		head_mask_SP=None,
		head_mask_RP=None,
		inputs_embeds_SP=None,
		inputs_embeds_RP=None,
		empathy_labels=None,
		rationale_labels=None,
		lambda_=0.1
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
			Labels for computing the token classification loss.
			Indices should be in ``[0, ..., config.num_labels - 1]``.
	Returns:
		:obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
		loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
			Classification loss.
		scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
			Classification scores (before SoftMax).
		hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
			Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
			of shape :obj:`(batch_size, sequence_length, hidden_size)`.
			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
			Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
			:obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
			heads.
	Examples::
		from transformers import RobertaTokenizer, RobertaForTokenClassification
		import torch
		tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
		model = RobertaForTokenClassification.from_pretrained('roberta-base')
		input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
		labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
		outputs = model(input_ids, labels=labels)
		loss, scores = outputs[:2]
		"""

		outputs_SP = self.seeker_encoder.roberta(
			input_ids_SP,
			attention_mask=attention_mask_SP,
			token_type_ids=token_type_ids_SP,
			position_ids=position_ids_SP,
			head_mask=head_mask_SP,
			inputs_embeds=inputs_embeds_SP,
		)


		outputs_RP = self.responder_encoder.roberta(
			input_ids_RP,
			attention_mask=attention_mask_RP,
			token_type_ids=token_type_ids_RP,
			position_ids=position_ids_RP,
			head_mask=head_mask_RP,
			inputs_embeds=inputs_embeds_RP,
		)


		sequence_output_SP = outputs_SP[0]
		sequence_output_RP = outputs_RP[0]


		SP_CLS = sequence_output_SP[:, 0, :]  # take <s> token (equiv. to [CLS])
		SP_CLS = self.dropout(SP_CLS)
		SP_CLS_h1 = self.ladder_hidden_SP_1(SP_CLS)
		# SP_CLS_h1 = torch.tanh(SP_CLS_h1)
		# SP_CLS_h2 = self.ladder_hidden_SP_2(SP_CLS_h1)


		RP_CLS = sequence_output_RP[:, 0, :]  # take <s> token (equiv. to [CLS])
		RP_CLS = self.dropout(RP_CLS)
		RP_CLS_h1 = self.ladder_hidden_RP_1(RP_CLS)
		# RP_CLS_h1 = torch.tanh(RP_CLS_h1)
		# RP_CLS_h2 = self.ladder_hidden_RP_2(RP_CLS_h1)

		concat_tensor = torch.cat([RP_CLS_h1, SP_CLS_h1], dim = -1)

		logits_empathy = self.empathy_classifier(torch.tanh(concat_tensor)) # (sequence_output_RP[:, 0, :]) #(torch.tanh(concat_tensor))

		# attend over seeker tokens 

		# attn_weights = F.softmax(
		#     self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		
		# attn_applied = torch.bmm(attn_weights.unsqueeze(0),
		#                          encoder_outputs.unsqueeze(0))


		sequence_output = self.dropout(sequence_output_RP)
		logits_rationales = self.rationale_classifier(sequence_output)


		outputs = (logits_empathy,logits_rationales) + outputs_RP[2:]


		if rationale_labels is not None:
			loss_fct = CrossEntropyLoss()
			# Only keep active parts of the loss
			if attention_mask_RP is not None:
				active_loss = attention_mask_RP.view(-1) == 1
				active_logits = logits_rationales.view(-1, self.rationale_num_labels)
				active_labels = torch.where(
					active_loss, rationale_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(rationale_labels)
				)
				loss_rationales = loss_fct(active_logits, active_labels)
			else:
				loss_rationales = loss_fct(logits_rationales.view(-1, self.rationale_num_labels), rationale_labels.view(-1))
			
			# outputs = (loss_rationales,) + outputs


		if empathy_labels is not None:
			loss_fct = CrossEntropyLoss()
			loss_empathy = loss_fct(logits_empathy.view(-1, self.empathy_num_labels), empathy_labels.view(-1))

			# outputs = (loss_empathy,) + outputs

		loss = loss_empathy + lambda_ * loss_rationales

		outputs = (loss, loss_empathy, loss_rationales) + outputs

		return outputs  # (loss), (scores_empathy, scores_rationales), (hidden_states), (attentions)





class BiEncoderAttentionWithRationaleClassification(nn.Module):

	# config_class = RobertaConfig
	# pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
	# base_model_prefix = "roberta"

	def __init__(self, hidden_dropout_prob=0.2, rationale_num_labels=2, empathy_num_labels=3, hidden_size=768, attn_heads = 1):
		super().__init__()

		# self.roberta_SP = RobertaModel(config)

		self.dropout = nn.Dropout(hidden_dropout_prob)
		
		self.rationale_classifier = nn.Linear(hidden_size, rationale_num_labels)

		# self.ladder_hidden_SP_1 = nn.Linear(hidden_size, 268)
		# self.ladder_hidden_SP_2 = nn.Linear(100, 100)
		
		# self.ladder_hidden_RP_1 = nn.Linear(hidden_size, 500)
		# self.ladder_hidden_RP_2 = nn.Linear(200, 200)

 
		self.attn = MultiHeadAttention(attn_heads, hidden_size)

		self.norm = Norm(hidden_size)


		self.rationale_num_labels = rationale_num_labels
		self.empathy_num_labels = empathy_num_labels

		self.empathy_classifier = RobertaClassificationHead(hidden_size = 768)

		self.apply(self._init_weights)

		self.seeker_encoder = SeekerEncoder.from_pretrained(
								"../../Empathy-Models/Pretraining-Tasks/pretrained-talklife-roberta-seeker/checkpoint-169000/", # Use the 12-layer BERT model, with an uncased vocab.
								output_attentions = False, # Whether the model returns attentions weights.
								output_hidden_states = False)

		self.responder_encoder = ResponderEncoder.from_pretrained(
								"../../Empathy-Models/Pretraining-Tasks/pretrained-talklife-roberta-response/checkpoint-293000/", # Use the 12-layer BERT model, with an uncased vocab.
								output_attentions = False, # Whether the model returns attentions weights.
								output_hidden_states = False)

	
	def _init_weights(self, module):
		""" Initialize the weights """
		if isinstance(module, (nn.Linear, nn.Embedding)):
			# Slightly different from the TF version which uses truncated_normal for initialization
			# cf https://github.com/pytorch/pytorch/pull/5617
			initializer_range=0.02
			module.weight.data.normal_(mean=0.0, std=initializer_range)
		elif isinstance(module, BertLayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()


	# @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
	def forward(
		self,
		input_ids_SP=None,
		input_ids_RP=None,
		attention_mask_SP=None,
		attention_mask_RP=None,
		token_type_ids_SP=None,
		token_type_ids_RP=None,
		position_ids_SP=None,
		position_ids_RP=None,
		head_mask_SP=None,
		head_mask_RP=None,
		inputs_embeds_SP=None,
		inputs_embeds_RP=None,
		empathy_labels=None,
		rationale_labels=None,
		lambda_=0.1
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
			Labels for computing the token classification loss.
			Indices should be in ``[0, ..., config.num_labels - 1]``.
	Returns:
		:obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
		loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
			Classification loss.
		scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
			Classification scores (before SoftMax).
		hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
			Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
			of shape :obj:`(batch_size, sequence_length, hidden_size)`.
			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
			Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
			:obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
			heads.
	Examples::
		from transformers import RobertaTokenizer, RobertaForTokenClassification
		import torch
		tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
		model = RobertaForTokenClassification.from_pretrained('roberta-base')
		input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
		labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
		outputs = model(input_ids, labels=labels)
		loss, scores = outputs[:2]
		"""

		outputs_SP = self.seeker_encoder.roberta(
			input_ids_SP,
			attention_mask=attention_mask_SP,
			token_type_ids=token_type_ids_SP,
			position_ids=position_ids_SP,
			head_mask=head_mask_SP,
			inputs_embeds=inputs_embeds_SP,
		)


		outputs_RP = self.responder_encoder.roberta(
			input_ids_RP,
			attention_mask=attention_mask_RP,
			token_type_ids=token_type_ids_RP,
			position_ids=position_ids_RP,
			head_mask=head_mask_RP,
			inputs_embeds=inputs_embeds_RP,
		)


		sequence_output_SP = outputs_SP[0]
		sequence_output_RP = outputs_RP[0]

		# RP_CLS = sequence_output_RP[:, 0, :]  # take <s> token (equiv. to [CLS])
		# RP_CLS = self.dropout(RP_CLS)

		# x2 = self.norm(sequence_output_RP)

		sequence_output_RP = sequence_output_RP + self.dropout(self.attn(sequence_output_RP, sequence_output_SP, sequence_output_SP))


		# RP_CLS = self.dropout(RP_CLS)


		# print(sequence_output_SP.size(), sequence_output_RP.size(), self.dropout(self.attn(sequence_output_RP[:, 0, :], sequence_output_SP[:, 0, :], sequence_output_SP[:, 0, :])).size(), RP_CLS.size())
		
		# exit(-1)
		logits_empathy = self.empathy_classifier(sequence_output_RP[:, 0, :]) # (sequence_output_RP[:, 0, :]) #(torch.tanh(concat_tensor))

		# attend over seeker tokens 

		# attn_weights = F.softmax(
		#     self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		
		# attn_applied = torch.bmm(attn_weights.unsqueeze(0),
		#                          encoder_outputs.unsqueeze(0))

		sequence_output = self.dropout(sequence_output_RP)
		
		# sequence_output = self.dropout(RP_all)
		logits_rationales = self.rationale_classifier(sequence_output)


		outputs = (logits_empathy,logits_rationales) + outputs_RP[2:]


		if rationale_labels is not None:
			loss_fct = CrossEntropyLoss()
			# Only keep active parts of the loss
			if attention_mask_RP is not None:
				active_loss = attention_mask_RP.view(-1) == 1
				active_logits = logits_rationales.view(-1, self.rationale_num_labels)
				active_labels = torch.where(
					active_loss, rationale_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(rationale_labels)
				)
				loss_rationales = loss_fct(active_logits, active_labels)
			else:
				loss_rationales = loss_fct(logits_rationales.view(-1, self.rationale_num_labels), rationale_labels.view(-1))
			
			# outputs = (loss_rationales,) + outputs


		if empathy_labels is not None:
			loss_fct = CrossEntropyLoss()
			loss_empathy = loss_fct(logits_empathy.view(-1, self.empathy_num_labels), empathy_labels.view(-1))

			# outputs = (loss_empathy,) + outputs

			loss = loss_empathy + lambda_ * loss_rationales

			outputs = (loss, loss_empathy, loss_rationales) + outputs

		return outputs  # (loss), (scores_empathy, scores_rationales), (hidden_states), (attentions)




class BiEncoderAttentionRationaleOnly(nn.Module):

	# config_class = RobertaConfig
	# pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
	# base_model_prefix = "roberta"

	def __init__(self, hidden_dropout_prob=0.2, rationale_num_labels=2, empathy_num_labels=3, hidden_size=768, attn_heads = 1):
		super().__init__()

		# self.roberta_SP = RobertaModel(config)

		self.dropout = nn.Dropout(hidden_dropout_prob)
		
		self.rationale_classifier = nn.Linear(hidden_size, rationale_num_labels)

		# self.ladder_hidden_SP_1 = nn.Linear(hidden_size, 268)
		# self.ladder_hidden_SP_2 = nn.Linear(100, 100)
		
		# self.ladder_hidden_RP_1 = nn.Linear(hidden_size, 500)
		# self.ladder_hidden_RP_2 = nn.Linear(200, 200)

 
		self.attn = MultiHeadAttention(attn_heads, hidden_size)

		self.norm = Norm(hidden_size)


		self.rationale_num_labels = rationale_num_labels
		self.empathy_num_labels = empathy_num_labels

		# self.empathy_classifier = RobertaClassificationHead(hidden_size = 768)

		self.apply(self._init_weights)

		self.seeker_encoder = SeekerEncoder.from_pretrained(
								"/projects/bdata/talklife/dssg/ashish/Codes/Empathy-Models/Pretraining-Tasks/pretrained-talklife-roberta-seeker/checkpoint-169000/", # Use the 12-layer BERT model, with an uncased vocab.
								output_attentions = False, # Whether the model returns attentions weights.
								output_hidden_states = False)

		self.responder_encoder = ResponderEncoder.from_pretrained(
								"/projects/bdata/talklife/dssg/ashish/Codes/Empathy-Models/Pretraining-Tasks/pretrained-talklife-roberta-response/checkpoint-293000/", # Use the 12-layer BERT model, with an uncased vocab.
								output_attentions = False, # Whether the model returns attentions weights.
								output_hidden_states = False)

	
	def _init_weights(self, module):
		""" Initialize the weights """
		if isinstance(module, (nn.Linear, nn.Embedding)):
			# Slightly different from the TF version which uses truncated_normal for initialization
			# cf https://github.com/pytorch/pytorch/pull/5617
			initializer_range=0.02
			module.weight.data.normal_(mean=0.0, std=initializer_range)
		elif isinstance(module, BertLayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()


	# @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
	def forward(
		self,
		input_ids_SP=None,
		input_ids_RP=None,
		attention_mask_SP=None,
		attention_mask_RP=None,
		token_type_ids_SP=None,
		token_type_ids_RP=None,
		position_ids_SP=None,
		position_ids_RP=None,
		head_mask_SP=None,
		head_mask_RP=None,
		inputs_embeds_SP=None,
		inputs_embeds_RP=None,
		empathy_labels=None,
		rationale_labels=None,
		lambda_=0.1
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
			Labels for computing the token classification loss.
			Indices should be in ``[0, ..., config.num_labels - 1]``.
	Returns:
		:obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
		loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
			Classification loss.
		scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
			Classification scores (before SoftMax).
		hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
			Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
			of shape :obj:`(batch_size, sequence_length, hidden_size)`.
			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
			Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
			:obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
			heads.
	Examples::
		from transformers import RobertaTokenizer, RobertaForTokenClassification
		import torch
		tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
		model = RobertaForTokenClassification.from_pretrained('roberta-base')
		input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
		labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
		outputs = model(input_ids, labels=labels)
		loss, scores = outputs[:2]
		"""

		outputs_SP = self.seeker_encoder.roberta(
			input_ids_SP,
			attention_mask=attention_mask_SP,
			token_type_ids=token_type_ids_SP,
			position_ids=position_ids_SP,
			head_mask=head_mask_SP,
			inputs_embeds=inputs_embeds_SP,
		)


		outputs_RP = self.responder_encoder.roberta(
			input_ids_RP,
			attention_mask=attention_mask_RP,
			token_type_ids=token_type_ids_RP,
			position_ids=position_ids_RP,
			head_mask=head_mask_RP,
			inputs_embeds=inputs_embeds_RP,
		)


		sequence_output_SP = outputs_SP[0]
		sequence_output_RP = outputs_RP[0]

		# RP_CLS = sequence_output_RP[:, 0, :]  # take <s> token (equiv. to [CLS])
		# RP_CLS = self.dropout(RP_CLS)

		# x2 = self.norm(sequence_output_RP)

		sequence_output_RP = sequence_output_RP + self.dropout(self.attn(sequence_output_RP, sequence_output_SP, sequence_output_SP))


		# RP_CLS = self.dropout(RP_CLS)


		# print(sequence_output_SP.size(), sequence_output_RP.size(), self.dropout(self.attn(sequence_output_RP[:, 0, :], sequence_output_SP[:, 0, :], sequence_output_SP[:, 0, :])).size(), RP_CLS.size())
		
		# exit(-1)
		# logits_empathy = self.empathy_classifier(sequence_output_RP[:, 0, :]) # (sequence_output_RP[:, 0, :]) #(torch.tanh(concat_tensor))

		# attend over seeker tokens 

		# attn_weights = F.softmax(
		#     self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		
		# attn_applied = torch.bmm(attn_weights.unsqueeze(0),
		#                          encoder_outputs.unsqueeze(0))

		sequence_output = self.dropout(sequence_output_RP)
		
		# sequence_output = self.dropout(RP_all)
		logits_rationales = self.rationale_classifier(sequence_output)


		outputs = (logits_rationales,) + outputs_RP[2:]


		if rationale_labels is not None:
			loss_fct = CrossEntropyLoss()
			# Only keep active parts of the loss
			if attention_mask_RP is not None:
				active_loss = attention_mask_RP.view(-1) == 1
				active_logits = logits_rationales.view(-1, self.rationale_num_labels)
				active_labels = torch.where(
					active_loss, rationale_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(rationale_labels)
				)
				loss_rationales = loss_fct(active_logits, active_labels)
			else:
				loss_rationales = loss_fct(logits_rationales.view(-1, self.rationale_num_labels), rationale_labels.view(-1))
			
			# outputs = (loss_rationales,) + outputs

			outputs = (loss_rationales,) + outputs

		return outputs  # (loss), (scores_rationales), (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
	"""Head for sentence-level classification tasks."""

	def __init__(self, hidden_dropout_prob=0.1, hidden_size=768, empathy_num_labels=3):
		super().__init__()

		self.dense = nn.Linear(hidden_size, hidden_size)
		self.dropout = nn.Dropout(hidden_dropout_prob)
		self.out_proj = nn.Linear(hidden_size, empathy_num_labels)

	def forward(self, features, **kwargs):
		x = features[:, :]  # take <s> token (equiv. to [CLS])
		x = self.dropout(x)
		x = self.dense(x)
		x = torch.relu(x)
		x = self.dropout(x)
		x = self.out_proj(x)
		return x



class GPT2SequenceClassifier(nn.Module):
	def __init__(self, hidden_size: int, num_classes:int):
		super(GPT2SequenceClassifier,self).__init__()
		
		self.gpt2model = GPT2Model.from_pretrained('gpt2')

		# GPT2ClassificationHead()
		self.num_labels = num_classes
		
		self.fc1 = nn.Linear(hidden_size, num_classes)
		
	def forward(self, x_in, labels):
		"""
		Args:
				x_in: encoded inputs ids of sent.
		"""
		
		gpt_out = self.gpt2model(x_in)[0] #returns tuple
		batch_size = gpt_out.shape[0]
		logits = self.fc1(gpt_out[:,-1,:]) #(batch_size , max_len, num_classes)

		outputs = (logits,)

		if labels is not None:
			if self.num_labels == 1:
				#  We are doing regression
				loss_fct = MSELoss()
				loss = loss_fct(logits.view(-1), labels.view(-1))
			else:
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			outputs = (loss,) + outputs

		return outputs 


class DialoGPTSequenceClassifier(nn.Module):
	def __init__(self, hidden_size: int, num_classes:int):
		super(DialoGPTSequenceClassifier,self).__init__()
		
		self.dialogpt_model = GPT2Model.from_pretrained("microsoft/DialoGPT-small")

		# GPT2ClassificationHead()

		self.num_labels = num_classes
		
		self.fc1 = nn.Linear(hidden_size, num_classes)
		
	def forward(self, x_in, labels):
		"""
		Args:
				x_in: encoded inputs ids of sent.
		"""
		
		gpt_out = self.dialogpt_model(x_in)[0] #returns tuple
		batch_size = gpt_out.shape[0]
		logits = self.fc1(gpt_out[:,-1,:]) #(batch_size , max_len, num_classes)

		outputs = (logits,)

		if labels is not None:
			if self.num_labels == 1:
				#  We are doing regression
				loss_fct = MSELoss()
				loss = loss_fct(logits.view(-1), labels.view(-1))
			else:
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			outputs = (loss,) + outputs

		return outputs


class GPT2ForTokenClassification(nn.Module):
	def __init__(self, hidden_size: int, num_classes:int):
		super(GPT2ForTokenClassification,self).__init__()
		
		self.gpt2model = GPT2Model.from_pretrained('gpt2')

		# GPT2ClassificationHead()
		self.num_labels = num_classes
		
		self.fc1 = nn.Linear(hidden_size, num_classes)
		
	def forward(self, x_in, attention_mask, labels):
		"""
		Args:
				x_in: encoded inputs ids of sent.
		"""
		
		gpt_out = self.gpt2model(x_in)[0] #returns tuple
		batch_size = gpt_out.shape[0]
		logits = self.fc1(gpt_out) #(batch_size , max_len, num_classes)

		outputs = (logits,)

		if labels is not None:
			loss_fct = CrossEntropyLoss()
			# Only keep active parts of the loss
			if attention_mask is not None:
				active_loss = attention_mask.view(-1) == 1
				active_logits = logits.view(-1, self.num_labels)
				active_labels = torch.where(
					active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
				)
				loss = loss_fct(active_logits, active_labels)
			else:
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			outputs = (loss,) + outputs


		return outputs 



class DialoGPTForTokenClassification(nn.Module):
	def __init__(self, hidden_size: int, num_classes:int):
		super(DialoGPTForTokenClassification,self).__init__()
		
		self.dialogpt_model = GPT2Model.from_pretrained("microsoft/DialoGPT-small")

		# GPT2ClassificationHead()
		self.num_labels = num_classes
		
		self.fc1 = nn.Linear(hidden_size, num_classes)
		
	def forward(self, x_in, attention_mask, labels):
		"""
		Args:
				x_in: encoded inputs ids of sent.
		"""
		
		gpt_out = self.dialogpt_model(x_in)[0] #returns tuple
		batch_size = gpt_out.shape[0]
		logits = self.fc1(gpt_out) #(batch_size , max_len, num_classes)

		outputs = (logits,)

		if labels is not None:
			loss_fct = CrossEntropyLoss()
			# Only keep active parts of the loss
			if attention_mask is not None:
				active_loss = attention_mask.view(-1) == 1
				active_logits = logits.view(-1, self.num_labels)
				active_labels = torch.where(
					active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
				)
				loss = loss_fct(active_logits, active_labels)
			else:
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			outputs = (loss,) + outputs
			
		return outputs 
