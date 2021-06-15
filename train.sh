python src/train_model.py \
	--model_name_or_path models/medium/ \
	--train_input_file dataset/sample_data.128len.db \
	--output_dir output/ \
	--log_dir output/ \
	--train_batch_size 4 \
	--num_optim_steps 500