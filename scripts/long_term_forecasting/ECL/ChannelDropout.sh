model_name=InvertedTransformer

python run.py \
    --is_training 1 \
    --model_id ${model_name}_ECL_ChannelDropout \
    --model ${model_name} \
    --data custom \
    --root_path ../dataset/electricity/ \
    --data_path electricity.csv \
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
    --patience 5 \
    --learning_rate 0.0005 \
    --use_norm 1 \