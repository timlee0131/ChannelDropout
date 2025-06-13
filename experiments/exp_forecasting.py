from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
import os
import time
import warnings
import numpy as np

from models import PatchTransformer
from utils.metrics import metric
from utils.tools import EarlyStopping

warnings.filterwarnings('ignore')

class Exp_Forecast():
    def __init__(self, args):
        self.args = args
        
        self.model_dict = {
            'PatchTransformer': PatchTransformer,
        }
        self.model = self._build_model().to(args.device)
    
    def _build_model(self):
        model = self.model_dict[self.args.model](self.args)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim
    
    def _select_scheduler(self):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=self.args.min_learning_rate)
        return scheduler
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        eval_criterion = nn.L1Loss()
        return criterion, eval_criterion
    
    def train(self):
        # data shape (B, L, N)
        train_data, train_loader = self._get_data('train')
        val_data, val_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')
        
        criterion, eval_criterion = self._select_criterion()
        optimizer = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, checkpointing=False)
        
        for epoch in range(self.args.epochs):
            self.model.train()
            
            epoch_loss = 0.0
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)
                
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.args.device)
                    batch_y_mark = batch_y_mark.float().to(self.args.device)
                
                y_hat = self.model(batch_x, batch_x_mark)
                loss = criterion(y_hat, batch_y)
                epoch_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            val_loss = self.val(val_loader, criterion)
            print(f"Epoch {epoch}, Loss (MSE): {epoch_loss / len(train_loader)}, Val Loss (MSE): {val_loss}, Time: {time.time() - epoch_time} sec")
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(optimizer, epoch+1, self.args)

        return self.model
    
    def val(self, val_loader, criterion):
        self.model.eval()
        
        total_loss = 0.0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)
                
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.args.device)
                    batch_y_mark = batch_y_mark.float().to(self.args.device)
                    
                y_hat = self.model(batch_x, batch_x_mark)
                total_loss += criterion(y_hat, batch_y).item()
                
        return total_loss / len(val_loader)
    
    def test(self):
        test_data, test_loader = self._get_data(flag='test')
        
        criterion, eval_criterion = self._select_criterion()
        
        total_loss_mse = 0.0
        total_loss_mae = 0.0
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)
                
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.args.device)
                    batch_y_mark = batch_y_mark.float().to(self.args.device)
                    
                y_hat = self.model(batch_x, batch_x_mark)
                total_loss_mse += criterion(y_hat, batch_y)
                total_loss_mae += eval_criterion(y_hat, batch_y)
                
        return total_loss_mse / len(test_loader), total_loss_mae / len(test_loader)
    
    def predict(self):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i, (batch_x, batch_x_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.args.device)
                
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.args.device)
                    
                y_hat = self.model(batch_x, batch_x_mark)
                preds.append(y_hat)
                
        return preds