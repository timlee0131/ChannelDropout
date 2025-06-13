model_name=PatchTransformer

python run.py \
    --is_training 1 \
    --model_id ${model_name}_Weather_ChannelDropout \
    --model ${model_name} \
    --data custom \
    --root_path ../dataset/weather/ \
    --data_path weather.csv \
    --features M \
    --batch_size 128 \
    --seq_len 96 \
    --pred_len 96 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.1 \
    --channel_dropout 0.1 \
    --num_layers 4 \
    --epochs 10 \
    --patience 5 \
    --learning_rate 0.0005 \
    --use_norm 1 \