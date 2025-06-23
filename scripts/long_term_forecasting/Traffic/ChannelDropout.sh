model_name=InvertedTransformer

python run.py \
    --is_training 1 \
    --model_id ${model_name}_Traffic_ChannelDropout \
    --model ${model_name} \
    --data custom \
    --root_path ../dataset/traffic/ \
    --data_path traffic.csv \
    --features M \
    --batch_size 4 \
    --seq_len 96 \
    --pred_len 96 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.1 \
    --channel_dropout 0.2 \
    --num_layers 4 \
    --epochs 10 \
    --patience 3 \
    --learning_rate 0.0001 \
    --use_norm 1 \
    --task forecast \
    --missing_rate 0.2 \
    --missing_pattern temporal_block \