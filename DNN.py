
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functools import partial
from typing import Tuple, Dict, List
from DataLoader import DataProcessor
from config import get_model_config
from processUtil import accuracyEvaluate
from tqdm import tqdm
from Timer import Timer

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------- 深度网络模块 ----------------------------- #
class DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes,output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.main = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            ResBlock(512, 256),
            ResBlock(256, 128),
            ResBlock(128, 64),
            ResBlock(64, 32),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return self.act(x + shortcut)

# ----------------------------- 训练器模块 ----------------------------- #
class DNNTrainer:
    def __init__(self,model_save_path: str, model , config: Dict):
        self.config = config
        self.writer = SummaryWriter()
        self.scaler = GradScaler()
        self.device = device
        self.model = model
        self.model_save_path = model_save_path
        self.is_train = True
        self.print_model_size()
        if os.path.exists(model_save_path):
            state_dict = torch.load(model_save_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            self.is_train = False


    def train(self, train_loader, val_loader):
        optimizer = self.config['optimizer'](self.model.parameters())
        scheduler = self.config['scheduler'](optimizer)
        best_loss = float('inf')
        if not self.is_train:
            return self.model

        for epoch in tqdm(range(self.config['epochs'])):
            self.model.train()
            train_loss = 0.0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                with autocast():
                    pred = self.model(x)
                    loss = self.config['loss_fn'](pred, y)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                train_loss += loss.item()
            
            train_loss = train_loss / len(train_loader)
            val_loss, y_pred = self.validation(val_loader)
            scheduler.step(val_loss)

            self.writer.add_scalar('Loss/Train', train_loss / len(train_loader), epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)

            print(f"Train epoch: {epoch:3d} -> Loss/Train: {train_loss:.4f}, Loss/Val: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_path)

        self.model.load_state_dict(torch.load(self.model_save_path))
        return self.model

    def validation(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                with autocast():
                    pred = self.model(x)
                    loss = self.config['loss_fn'](pred, y)
                total_loss += loss.item()
        total_loss = total_loss / len(loader)
        return total_loss, np.hstack((y.cpu().numpy().reshape(-1,1),pred.cpu().numpy().reshape(-1,1)))
    
    def save_results_and_evaluate(self, outputs: np.ndarray, save_path: str, description: str):
        """保存预测结果并进行评估"""
        np.savetxt(save_path, outputs, delimiter=',',fmt='%s')
        R, BIAS, MAE, MRE, RMSE, KGE = accuracyEvaluate(outputs[:, -2], outputs[:, -1], False)
        print(f"{description} Evaluation -> R: {R:.4f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, KGE: {KGE:.4f}")
    
    def print_model_size(self) -> None:
        """Print the number of trainable parameters in the initialized NN model."""
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized successfully with the number of trainable parameters: {num_params:,}")

# ----------------------------- 辅助函数 ----------------------------- #
def create_data_loaders(X, y, batch_size=256, val_size=0.2):
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1))
    train_loader, val_loader = None,None
    if val_size > 0:
        train_size = int((1 - val_size) * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=val_size)
    else:
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# ----------------------------- 主函数 ----------------------------- #
DistanceDensity = ['all','20km','40km','60km','80km','100km','120km','140km','160km','180km','200km','220km','240km','260km','280km','300km']
def main():
    config0 = {
        'epochs': 100,
        'batch_size': 64,
        'loss_fn': nn.MSELoss(),
        'optimizer': partial(optim.AdamW, lr=1e-3),
        'scheduler': partial(optim.lr_scheduler.ReduceLROnPlateau, patience=5)
    }

    wp_path = './'
    channel = 'T3'
    cmask = 'cloud'

    elapsed_list = []
    for Dis in DistanceDensity:
        print(f'starting processing {Dis} ...')
        test_result_save_path = os.path.join(wp_path, 'result', f'DNN_{channel}_{cmask}_{Dis}_test.csv')
        Val_result_save_path = os.path.join(wp_path, 'result', f'DNN_{channel}_{cmask}_{Dis}_cv10.csv')

        data_processor = DataProcessor(
            train_file=os.path.join(wp_path, 'data', 'dataset', f'train_{cmask}_{Dis}.csv'),
            test_file=os.path.join(wp_path, 'data','dataset', f'test_{cmask}.csv'),
            channel=channel
        )
        data_processor.load_data()
        
        ### 训练集和验证集
        train_loader, val_loader = create_data_loaders(data_processor.X_train_scaled, data_processor.y_train, batch_size=config0['batch_size'], val_size=0.2)
        ### 测试集
        _, test_loader = create_data_loaders(data_processor.X_test_scaled, data_processor.y_test, batch_size=len(data_processor.y_test), val_size=0.0)

        model = DNN(input_size=data_processor.X_train_scaled.shape[1],hidden_sizes=[28],output_size=1).to(device)

        trainer = DNNTrainer(model_save_path=os.path.join(wp_path, 'model', f'DNN_best_{channel}_{cmask}_{Dis}.pth'),
                                model=model,
                                config=config0
                                )
        timer = Timer("begining...", verbose=False)
        timer.start()
        model = trainer.train(train_loader, val_loader)
        elapsed = timer.stop()
        print(f"{Dis} elapsed: {elapsed:.6f} s")
        elapsed_list.append([Dis,elapsed])

        mse_val, y_pred_val= trainer.validation(val_loader)
        trainer.save_results_and_evaluate(y_pred_val, Val_result_save_path, "CV5 Set")
        mse_test, y_pred_test = trainer.validation(test_loader)
        test_outs = np.hstack((data_processor.X_test[:, np.r_[0:4, -3:]], y_pred_test))
        trainer.save_results_and_evaluate(test_outs, test_result_save_path, "Test Set")
    
    elapsed_list = np.array(elapsed_list)
    np.savetxt(os.path.join(wp_path, 'model', 'elapsed', f'DNN_{channel}_{cmask}_elapsed.csv'),elapsed_list, delimiter=',',fmt='%s')
    print('All done!')


if __name__ == '__main__':
    main()
