import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import os

class DataProcessor:
    """负责数据读取和预处理"""

    def __init__(self, train_file, test_file,channel='T3'):
        self.train_file = train_file
        self.test_file = test_file
        self.channel = channel
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler()
        # self.scaler = MinMaxScaler()
        # self.scaler_y = MinMaxScaler()

        self.X_train = None
        self.X_train_scaled = None
        self.y_train = None
        self.y_train_scaled = None

        self.X_test = None
        self.X_test_scaled = None
        self.y_test = None
        self.y_test_scaled = None

    def _load_and_process(self, file_path):
        dataset = np.genfromtxt(file_path, delimiter=',')
        filename = os.path.basename(file_path)
        isClear = False
        if "clear" in filename:
            isClear = True

        lat,lon,h,time,lc,cmask,Band02_rf,Band05_rf,Band17_rf,Band18_rf,Band19_rf,T_17_2cr,T_18_2cr,T_19_2cr,T_17_3cr,T_18_3cr,T_19_3cr,SensorZ,SolarZ,Htop,Ttop,Tsur,Ptop,Psur,PWV_star,PWVmod,PWVgnss = \
        dataset[:,0],dataset[:,1],dataset[:,2],dataset[:,3],dataset[:,4],dataset[:,5],dataset[:,6],dataset[:,7],dataset[:,8],dataset[:,9],dataset[:,10],\
        dataset[:,11],dataset[:,12],dataset[:,13],dataset[:,14],dataset[:,15],dataset[:,16],dataset[:,17],dataset[:,18],dataset[:,21],dataset[:,23],dataset[:,24],dataset[:,25],dataset[:,26],\
        dataset[:,29],dataset[:,30],dataset[:,31]
        DOY = np.floor(time)
        HOD = (time-DOY)*60
        if self.channel == 'T3':
            features = np.column_stack((lat,lon,h,DOY,HOD,lc,SensorZ,SolarZ,T_17_3cr,T_18_3cr,T_19_3cr,Ptop,Ttop,Htop))
        elif self.channel == 'T2':
            features = np.column_stack((lat,lon,h,DOY,HOD,lc,SensorZ,SolarZ,T_17_2cr,T_18_2cr,T_19_2cr,Ptop,Ttop,Htop))

        if isClear:
            features = features[:,:-3]
        targets = PWVgnss
        mask = np.isfinite(features).all(axis=1) & np.isfinite(targets)
        features = features[mask]
        targets = targets[mask]
        features = np.nan_to_num(features)
        targets = np.nan_to_num(targets)

        return features, targets

    def load_data(self):
        self.X_train, self.y_train = self._load_and_process(self.train_file)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        self.X_test, self.y_test = self._load_and_process(self.test_file)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.y_test_scaled = self.scaler_y.transform(self.y_test.reshape(-1, 1)).flatten()
    
    def inverse_transform_X(self, X_scaled):
        X_inv = self.scaler.inverse_transform(X_scaled)
        return X_inv
    
    def inverse_transform_y(self, y_scaled):
        y_inv = self.scaler_y.inverse_transform(y_scaled)
        return y_inv
    
    def inverse_transform_multiy(self, y_scaled_array):
        X_inversed = np.zeros_like(y_scaled_array)
        for i in range(y_scaled_array.shape[1]):
            X_inversed[:, i] = self.scaler_y.inverse_transform(y_scaled_array[:, i].reshape(-1, 1)).ravel()
        return X_inversed
    
    def remove_outliers(self):
        pass