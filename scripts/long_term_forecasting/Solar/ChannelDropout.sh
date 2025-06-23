model_name=InvertedTransformer

python run.py \
    --is_training 1 \
    --model_id ${model_name}_Solar_ChannelDropout \
    --model ${model_name} \
    --data Solar \
    --root_path ../dataset/solar/ \
    --data_path solar_AL.txt \
    --features M \
    --batch_size 128 \
    --seq_len 96 \
    --pred_len 96 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.1 \
    --channel_dropout 0.2 \
    --num_layers 6 \
    --epochs 10 \
    --patience 5 \
    --learning_rate 0.0001 \
    --use_norm 1 \