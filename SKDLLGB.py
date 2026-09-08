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
from config import DNN_CONFIG, SKDL_CONFIG

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------- 特征工程模块 ----------------------------- #
class GBMTFeatureGenerator:
    def __init__(self, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        self.scaler = StandardScaler()
        self.lgb_model = None
        self.optimal_params = None
        self.num_leaves = None

    def auto_tune(self, X: np.ndarray, y: np.ndarray, n_trials: int = 20):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 30, 200),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0)
            }
            leaves_train, leaves_val = self.train_gbmt(X_train, y_train, X_val, params)
            model = self._create_model(X.shape[1], params['n_estimators'], params['num_leaves'])
            return self._evaluate_model(model, leaves_train, y_train, leaves_val, y_val, X_train, X_val)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        self.optimal_params = study.best_params
        self.train_gbmt(X, y, None, self.optimal_params)

    def train_gbmt(self, X_train, y_train, X_val=None, params=None):
        if params is not None:
            self.optimal_params = params
        X_scaled = self.scaler.fit_transform(X_train)
        lgb_train = lgb.Dataset(X_scaled, y_train)

        self.lgb_model = lgb.train(params, lgb_train, num_boost_round=100)
        self.num_leaves = params['num_leaves']

        leaves_train = self.lgb_model.predict(X_scaled, pred_leaf=True)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            leaves_val = self.lgb_model.predict(X_val_scaled, pred_leaf=True)
            return leaves_train, leaves_val
        return leaves_train, None

    def _create_model(self, num_numerical, n_estimators, num_leaves):
        return SKDLLGB(num_numerical=num_numerical, n_estimators=n_estimators, num_leaves=num_leaves).to(device)

    def _evaluate_model(self, model, leaves_train, y_train, leaves_val, y_val, X_train, X_val):
        X_num_train = torch.FloatTensor(self.scaler.transform(X_train)).to(device)
        X_emb_train = torch.LongTensor(leaves_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)

        train_loader = DataLoader(TensorDataset(X_num_train, X_emb_train, y_train_tensor), batch_size=256, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(5):
            for x_num, x_emb, y in train_loader:
                optimizer.zero_grad()
                pred = model(x_num, x_emb)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_num_val = torch.FloatTensor(self.scaler.transform(X_val)).to(device)
            X_emb_val = torch.LongTensor(leaves_val).to(device)
            preds = model(X_num_val, X_emb_val).cpu().numpy()
            return np.sqrt(np.mean((y_val - preds.squeeze())**2))

    def generate_features(self, X: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        X_scaled = self.scaler.transform(X)
        leaves = self.lgb_model.predict(X_scaled, pred_leaf=True)
        return torch.FloatTensor(X_scaled).to(self.device), torch.LongTensor(leaves).to(self.device)

# ----------------------------- 深度网络模块 ----------------------------- #
class SKDLLGB(nn.Module):
    def __init__(self, num_numerical, n_estimators, num_leaves, emb_dim=SKDL_CONFIG['embedding_dim'], use_attention=SKDL_CONFIG['use_attention']):
        super().__init__()
        self.emb = nn.Embedding(n_estimators * num_leaves, emb_dim)
        # self.use_attention = use_attention
        # self.attention = nn.MultiheadAttention(emb_dim, 4, batch_first=True) if use_attention else None

        self.main = nn.Sequential(
            nn.Linear(num_numerical + emb_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            ResBlock(1024, 512),
            ResBlock(512, 256),
            ResBlock(256, 128),
            ResBlock(128, 64),
            ResBlock(64, 32),
            nn.Linear(32, 1)
        )

    def forward(self, x_num, x_emb):
        emb = self.emb(x_emb).view(x_emb.size(0), -1, self.emb.embedding_dim)
        # if self.use_attention:
        #     emb, _ = self.attention(emb, emb, emb)
        emb = emb.mean(dim=1)
        x = torch.cat([x_num, emb], dim=1)
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
class SKDLLGBTrainer:
    def __init__(self,model_save_path: str, model , config: Dict):
        self.config = config
        self.writer = SummaryWriter()
        self.scaler = GradScaler()
        self.device = device
        self.model = model
        self.model_save_path = model_save_path
        self.print_model_size()
        if os.path.exists(model_save_path):
            state_dict = torch.load(model_save_path, map_location=device)
            model.load_state_dict(state_dict)

        model.to(device)


    def train(self, train_loader, val_loader):
        optimizer = self.config['optimizer'](self.model.parameters())
        scheduler = self.config['scheduler'](optimizer)
        best_loss = float('inf')

        for epoch in tqdm(range(self.config['epochs'])):
            self.model.train()
            train_loss = 0.0

            for x_num, x_emb, y in train_loader:
                x_num, x_emb, y = x_num.to(self.device), x_emb.to(self.device), y.to(self.device)

                with autocast():
                    pred = self.model(x_num, x_emb)
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
            for x_num, x_emb, y in loader:
                x_num, x_emb, y = x_num.to(self.device), x_emb.to(self.device), y.to(self.device)
                with autocast():
                    pred = self.model(x_num, x_emb)
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
def create_data_loaders(X_num, X_emb, y, batch_size=256, val_size=0.2):
    dataset = TensorDataset(X_num, X_emb, torch.FloatTensor(y).unsqueeze(1))
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
# DistanceDensity = ['all','20km','40km','60km','80km','100km','120km','140km','160km','180km','200km','220km','240km','260km','280km','300km']
DistanceDensity = ['all']
def main():
    config0 = DNN_CONFIG

    wp_path = './'
    channel = 'T3'
    cmask = 'cloud'
    model_name = 'lightgbm'

    elapsed_list = []
    for Dis in DistanceDensity:
        print(f'starting processing {Dis} ...')
        test_result_save_path = os.path.join(wp_path, 'result', f'deepGBM_{channel}_{cmask}_{Dis}_test.csv')
        Val_result_save_path = os.path.join(wp_path, 'result', f'deepGBM_{channel}_{cmask}_{Dis}_cv10.csv')

        data_processor = DataProcessor(
            train_file=os.path.join(wp_path, 'data', 'dataset', f'train_{cmask}_{Dis}.csv'),
            test_file=os.path.join(wp_path, 'data','dataset', f'test_{cmask}.csv'),
            channel=channel
        )
        data_processor.load_data()
        best_params = get_model_config(model_name)
        feature_gen = GBMTFeatureGenerator()
        feature_gen.train_gbmt(data_processor.X_train, data_processor.y_train, X_val=None, params=best_params)

        
        ### 测试集
        X_num, X_emb = feature_gen.generate_features(data_processor.X_test)
        _, test_loader = create_data_loaders(X_num, X_emb, data_processor.y_test, batch_size=len(data_processor.y_test), val_size=0.0)

        ### 训练集和验证集
        X_num, X_emb = feature_gen.generate_features(data_processor.X_train)
        train_loader, val_loader = create_data_loaders(X_num, X_emb, data_processor.y_train, batch_size=config0['batch_size'], val_size=0.2)

        model = SKDLLGB(
            num_numerical=X_num.shape[1],
            n_estimators=feature_gen.optimal_params['n_estimators'],
            num_leaves=feature_gen.num_leaves
        ).to(device)

        trainer = SKDLLGBTrainer(model_save_path=os.path.join(wp_path, 'model', f'deepGBM_best_{channel}_{cmask}_{Dis}.pth'),
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
    np.savetxt(os.path.join(wp_path, 'model', 'elapsed', f'deepGBM_{channel}_{cmask}_elapsed.csv'),elapsed_list, delimiter=',',fmt='%s')
    print('All done!')

if __name__ == '__main__':
    main()
