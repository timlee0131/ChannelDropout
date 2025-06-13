import argparse
import torch
from experiments.exp_forecasting import Exp_Forecast
import random
import numpy as np
import time

if __name__ == "__main__":
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='ChannelDropout')
    
    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='ChannelDropout', help='model name, options: [ChannelDropout]')
    
    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    
    # forecasting config
    parser.add_argument('--num_patches', type=int, default=6, help='number of patches')
    parser.add_argument('--seq_len', type=int, default=96, help='input (lookback) window length')
    parser.add_argument('--label_len', type=int, default=0, help='overlap length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction (horizon) length')
    
    # model config
    parser.add_argument('--d_input', type=int, default=1, help='input dimension')
    parser.add_argument('--d_model', type=int, default=128, help='model hidden dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    
    # optimization
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--channel_dropout', type=float, default=0.01, help='channel dropout rate')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='optimizer weight decay')
    
    # misc.
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--sampling_rate', type=int, default=1.0, help='sampling rate')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args.device = device
    
    print("Args in experiment:")
    print(args)
    
    exp = Exp_Forecast(args)
    
    if args.is_training:
        model = exp.train()
        test_loss_mse, test_loss_mae = exp.test()
        print(f"\nTest Loss (MSE): {test_loss_mse}, Test Loss (MAE): {test_loss_mae}")
    else:
        test_loss_mse, test_loss_mae = exp.test()
        print(f"Test Loss (MSE): {test_loss_mse}, Test Loss (MAE): {test_loss_mae}")
    
    