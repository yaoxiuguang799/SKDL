#!/usr/bin/env python
# coding: utf-8

import os
import math
import warnings
import numpy as np
import joblib
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from processUtil import accuracyEvaluate
from DataLoader import DataProcessor
from config import get_model_config
from Timer import Timer

warnings.filterwarnings('ignore')

class RandomForestBayesTrainer:
    """负责模型训练、贝叶斯优化、评估、保存"""

    def __init__(self, data_processor: DataProcessor, model_save_path: str, result_save_path: str, param_bounds: dict = None, random_state: int = 7):
        self.data = data_processor
        self.model_save_path = model_save_path
        self.test_result_save_path = result_save_path
        self.cv5_result_save_path = result_save_path.replace("test", "cv10", 1)
        self.random_state = random_state
        self.model_name = 'randomforest'
        self.param_bounds = param_bounds or {
            'n_estimators': (10, 600),
            'max_depth': (10, 100),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 4)
        }
        self.model = None
        self.load_model()

    def _build_model(self, params: dict) -> RandomForestRegressor:
        """基于参数构建XGBoost模型"""
        return RandomForestRegressor(
            n_estimators=math.ceil(params['n_estimators']),
            max_depth=math.ceil(params['max_depth']),
            min_samples_split=math.ceil(params['min_samples_split']),
            min_samples_leaf=math.ceil(params['min_samples_leaf']),
            random_state=self.random_state
        )

    def bayes_opt_objective(self, n_estimators, max_depth, min_samples_split, min_samples_leaf):
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }
        model = self._build_model(params)
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_validate(
            model, self.data.X_train, self.data.y_train,
            scoring='neg_mean_squared_error',
            cv=cv, n_jobs=-1, error_score='raise'
        )
        return np.mean(scores['test_score'])

    def save_model(self):
        """保存模型到磁盘"""
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        joblib.dump(self.model, self.model_save_path)

    def load_model(self):
        """尝试加载已有模型"""
        if os.path.exists(self.model_save_path):
            self.model = joblib.load(self.model_save_path)

    def optimize_hyperparameters(self, init_points=10, n_iter=10):
        """使用贝叶斯优化寻找超参数"""
        optimizer = BayesianOptimization(
            f=self.bayes_opt_objective,
            pbounds=self.param_bounds,
            random_state=self.random_state
        )
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        return optimizer.max['params'], optimizer.max['target']

    def fit_best_model(self, params: dict):
        """训练最佳模型，并返回交叉验证集预测结果"""
        self.model = self._build_model(params)
        X_train, X_val, y_train, y_val = train_test_split(self.data.X_train, self.data.y_train, test_size=0.2, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        val_pred = self.model.predict(X_val)
        predictions = np.hstack((X_val[:, np.r_[0:4, -3:]], y_val.reshape(-1, 1), val_pred.reshape(-1, 1)))
        return predictions

    def validate(self):
        """在测试集上验证模型，并返回预测结果"""
        pred_test = self.model.predict(self.data.X_test)
        predictions = np.hstack((self.data.X_test[:, np.r_[0:4, -3:]], self.data.y_test.reshape(-1, 1), pred_test.reshape(-1, 1)))
        return predictions

    def _save_results_and_evaluate(self, outputs: np.ndarray, save_path: str, description: str):
        """保存预测结果并进行评估"""
        np.savetxt(save_path, outputs, delimiter=',',fmt='%s')
        R, BIAS, MAE, MRE, RMSE, KGE = accuracyEvaluate(outputs[:, -2], outputs[:, -1], False)
        print(f"{description} Evaluation -> R: {R:.4f}, MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, KGE: {KGE:.4f}")

    def run(self):
        """完整训练流程：超参搜索、模型训练、验证、保存"""
        print("Loading and processing data...")
        self.data.load_data()

        print("Starting hyperparameter optimization...")
        best_params, best_score = self.optimize_hyperparameters()
        print(f'Best training score (neg MSE): {abs(best_score):.4f}')

        print("Fitting best model...")
        cv5_val_outs = self.fit_best_model(best_params)
        self._save_results_and_evaluate(cv5_val_outs, self.cv5_result_save_path, "CV5 Validation")

        print("Validating model using test dataset...")
        test_outs = self.validate()
        self._save_results_and_evaluate(test_outs, self.test_result_save_path, "Test Set")

        print("Saving model...")
        self.save_model()
        print("Training complete!")

    def fit(self):
        """直接用已有配置参数训练模型并评估"""
        print("Loading and processing data...")
        self.data.load_data()

        best_params = get_model_config(self.model_name)
        print("Fitting best model...")
        cv5_val_outs = self.fit_best_model(best_params)
        self._save_results_and_evaluate(cv5_val_outs, self.cv5_result_save_path, "CV5 Validation")

        print("Validating model using test dataset...")
        test_outs = self.validate()
        self._save_results_and_evaluate(test_outs, self.test_result_save_path, "Test Set")
        
        print("Saving model...")
        self.save_model()
        print("Fitting complete!")

    def test(self):
        """仅用已有模型进行测试集评估"""
        print("Loading and processing data...")
        self.data.load_data()

        print("Validating model using test dataset...")
        test_outs = self.validate()
        self._save_results_and_evaluate(test_outs, self.test_result_save_path, "Test Set")

        print("Testing complete!")

DistanceDensity = ['all','20km','40km','60km','80km','100km','120km','140km','160km','180km','200km','220km','240km','260km','280km','300km']
DistanceDensity = ['all']
if __name__ == '__main__':
    wp_path = './'
    channel = 'T3'
    cmask = 'cloud'
    elapsed_list = []
    for Dis in DistanceDensity:
        print(f'starting processing {Dis} ...')
        data_processor = DataProcessor(
            train_file=os.path.join(wp_path, 'data', 'dataset', f'train_{cmask}_{Dis}.csv'),
            test_file=os.path.join(wp_path, 'data','dataset', f'test_{cmask}.csv'),
            channel=channel
        )

        trainer = RandomForestBayesTrainer(
            data_processor=data_processor,
            model_save_path=os.path.join(wp_path, 'model', f'RandomForest_best_{channel}_{cmask}_{Dis}.joblib'),
            result_save_path=os.path.join(wp_path, 'result', f'RandomForest_{channel}_{cmask}_{Dis}_test.csv')
        )

        timer = Timer("begining...", verbose=False)
        timer.start()
        trainer.run()
        elapsed = timer.stop()
        print(f"{Dis} elapsed: {elapsed:.6f} s")
        elapsed_list.append([Dis,elapsed])
    elapsed_list = np.array(elapsed_list)
    np.savetxt(os.path.join(wp_path, 'model', 'elapsed', f'RandomForest_{channel}_{cmask}_elapsed.csv'),elapsed_list, delimiter=',',fmt='%s')

    print('All done!')
