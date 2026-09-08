def get_model_config(model_type, X_train=None,y_train=None):
    if model_type == "xgboost":
        return {
            'objective': 'reg:squarederror',  
            'grow_policy': 'lossguide',       
            'max_leaves': 77,                 
            'n_estimators': 485,
            'max_depth': 28,
            'learning_rate': 0.088,
            'min_child_weight': 14
        }
    if model_type == "lightgbm":
        return {
            'n_estimators': 319,
            'max_depth': 21,
            'learning_rate': 0.2,
            'num_leaves': 77
        }
    elif model_type == "randomforest":
        return {
            'n_estimators': 538,
            'max_depth': 58,
            'min_samples_split': 4,
            'min_samples_leaf': 1
        }
    elif model_type == "deepforest":
        return {
            'n_estimators': 8,
            'n_trees': 100,
            'max_layers': 3,
            'max_depth ': None
        }
    elif model_type == "bpnn":
        input_size = X_train.shape[1]
        hidden_sizes = input_size*2 + 1 - 3
        return {
            'input_size': X_train.shape[1],
            'hidden_sizes': [hidden_sizes],
            'lr': 0.001,
            'epochs': 100,
            'batch_size': 64
        }
    elif model_type == "grnn":
        return {
            'sigma': 0.05,
            'X_train': X_train,
            'y_train': y_train
        }
    else:
        raise ValueError(f"Unsupported model: {model_type}")