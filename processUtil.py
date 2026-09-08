import numpy as np
from scipy.stats import pearsonr
import scipy
import h5py
import os
from scipy.io import savemat,loadmat
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
import rasterio
from affine import Affine
from scipy.ndimage import map_coordinates

def extract_parent_dir(path: str) -> str:
    """Extract the given path's parent directory.

    Parameters
    ----------
    path :
        The path for extracting.

    Returns
    -------
    parent_dir :
        The path to the parent dir of the given path.

    """
    parent_dir = os.path.abspath(os.path.join(path, ".."))
    return parent_dir
def create_dir_if_not_exist(path: str, is_dir: bool = True) -> None:
    """Create the given directory if it doesn't exist.

    Parameters
    ----------
    path :
        The path for check.

    is_dir :
        Whether the given path is to a directory. If `is_dir` is False, the given path is to a file or an object,
        then this file's parent directory will be checked.

    """
    
    path = extract_parent_dir(path) if not is_dir else path
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
#判断闰年
def is_leap_year(year):
    return  (year % 4 == 0 and year % 100 != 0) or year % 400 == 0

def date2doy(year,month,day):
    month_leapyear=[31,29,31,30,31,30,31,31,30,31,30,31]
    month_notleap= [31,28,31,30,31,30,31,31,30,31,30,31]
    doy=0

    if month==1:
            pass
    elif is_leap_year(year):
            for i in range(month-1):
                    doy+=month_leapyear[i]
    else:
            for i in range(month-1):
                    doy+=month_notleap[i]
    doy+=day
    return doy
def read_gnss_pwv_bydaily(files):
    doys = []
    blpwvs = []
    for file in files:
        basename = os.path.basename(file)
        data = basename.split('.')
        doy = int(data[1])
        blpwv = []
        with open(file,'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.split()
                b = float(data[1])
                l = float(data[2])
                pwv = float(data[4])
                if pwv == -9.9:
                    pwv = np.nan 
                blpwv.append([b,l,pwv])
        if len(blpwv) != 0:
            blpwv = np.array(blpwv)
            doys.append(doy)
            blpwvs.append(blpwv)
    
    return doys,blpwvs

def read_gnss_pwv(year):
    gnss_pwv_file = '.\\GNSS_pwv\\GNSS_PWV.mat'
    gnss_pwv_geolocation = '.\\GNSS_pwv\\cmonoc_station_PPP.txt'
    
    sites = []
    with open(gnss_pwv_geolocation,'r') as f:
        lines = f.readlines()
        num_site = len(lines)
        for line in lines:
            sites.append(line.split())
    data = loadmat(gnss_pwv_file)
    datas = data['GNSS_PWV_PPP']
    nday = 365
    if is_leap_year(year):
        nday = 366
    
    gnss_pwvs = np.full((num_site,nday),np.nan)
    site_name,lat,lon = [],[],[]
    for isite in range(num_site):
        site_name.append(sites[isite][0])
        lat.append(float(sites[isite][2]))
        lon.append(float(sites[isite][1]))
        pwv_by_site = datas[isite][0]
        ind = np.where((pwv_by_site[:,0] == year) & ((pwv_by_site[:,2] == 3.0) | (pwv_by_site[:,2] == 4.0) | (pwv_by_site[:,2] == 5.0) | (pwv_by_site[:,2] == 6.0)))
        pwvs = pwv_by_site[ind]
        uniqueList = np.unique(pwvs[:,1])
        for doy in uniqueList:
            idoy = int(doy - 1)
            ind = np.where(pwvs[:,1] == doy)
            pwv_tmp = pwvs[ind]
            ipwv = np.mean(pwv_tmp[:,3])
            gnss_pwvs[isite,idoy] = ipwv

    return site_name,lat,lon,gnss_pwvs
def read_gnss_pwv_usa(file,minlat,maxlat,minlon,maxlon):
    pwv = []
    doy = []
    with open(file,'r') as f:
        blhLine = f.readline()
        tmp = blhLine.split()
        b = float(tmp[1])
        l = float(tmp[2])
        h = float(tmp[3])
        if b < minlat or b > maxlat or l < minlon or l > maxlon:
            return b,l,h,[],[]
        lines = f.readlines()
        for line in lines:
            buff = line.split()
            tmp = buff[0].split('-')
            year = int(tmp[0])
            month = int(tmp[1])
            day = int(tmp[2])
            doyi = date2doy(year,month,day)
            if buff[2] == 'nan' or buff[2] == '-9.9':
                pwvi = np.nan
            else:
                pwvi = float(buff[2])
            pwv.append(pwvi)
            doy.append(doyi)
        return b,l,h,doy,pwv
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

def search_window(irow_m,icol_m,nrow,ncol,w=3):
    h = int(w/2)
    urow_w = irow_m + h + 1
    drow_w = irow_m - h
    if drow_w < 0:
        drow_w = 0
        urow_w = drow_w + w
    if urow_w > nrow - 1:
        urow_w = nrow - 1
        drow_w = urow_w - w
    
    lcol_w = icol_m - h 
    rcol_w = icol_m + h + 1
    if lcol_w < 0 :
        lcol_w = 0
        rcol_w = lcol_w + w
    if rcol_w > ncol - 1:
        rcol_w = ncol - 1
        lcol_w = rcol_w - w

    return drow_w,urow_w,lcol_w,rcol_w

def search_nearby_gridpoints(point_lat,ponit_lon,grid_lat, grid_lon,grid):
    max_row = len(grid_lat);max_col = len(grid_lon)
    row,nearest_lat = find_nearest(grid_lat, point_lat)
    col,nearest_lon = find_nearest(grid_lon, ponit_lon )
    lat_near = []
    lon_near = []
    grid_near = []
    # 搜索附近9个点
    for r in range(row-1, row+2):
        for c in range(col-1, col+2):
            if 0 <= r < max_row and 0 <= c < max_col:
                lat_near.append(grid_lat[r])
                lon_near.append(grid_lon[c])
                grid_near.append(grid[r,c])
    
    return np.array(lat_near),np.array(lon_near),np.array(grid_near)
     

def select_similar_points(idoy,drow_w,lcol_w,irow_m,icol_m,pwv_w,imask_w,pwv_avc_sereis_m,avc_pwvs_w):
    row_w, col_w = np.where((imask_w==1) | (imask_w==2)) # imask_w==1 表示modis有数据，imask_w==2表示ERA5有数据
    num_w = len(row_w)
    pwv_fus = []
    pwv_avc = []
    Q_dist = []
    for ind in range(num_w):
        irow_w = row_w[ind]
        icol_w = col_w[ind]
        pwv_avc_sereis_s = avc_pwvs_w[:,irow_w,icol_w]
        
        # 把搜索到的点在搜索窗口中的位置转换至大矩阵中的位置
        irow_w_in_area = drow_w + irow_w
        icol_w_in_area = lcol_w + icol_w
        # 计算搜索到的点至m点的距离,并计算反距离加权
        drow = irow_w_in_area - irow_m
        dcol = icol_w_in_area - icol_m
        # 获取搜索到的点的avc模型参数
        if drow == 0 and  dcol == 0:
            continue
        q_idist = 1/np.sqrt(drow*drow+dcol*dcol)

        mse = np.sum(np.square(pwv_avc_sereis_m-pwv_avc_sereis_s))/len(pwv_avc_sereis_s)
        RMSE = np.sqrt(mse)  #均方根误差
        if RMSE < 100:
            pwv_fus.append(pwv_w[irow_w,icol_w])
            pwv_avc.append(avc_pwvs_w[idoy,irow_w,icol_w])
            Q_dist.append(q_idist)
    pwv_fus = np.array(pwv_fus);pwv_avc = np.array(pwv_avc);Q_dist = np.array(Q_dist)
    # 粗差探测
    if len(pwv_avc) >= 3:
        ind = out_range_method(pwv_fus-pwv_avc)
        pwv_fus = np.delete(pwv_fus,ind)
        pwv_avc = np.delete(pwv_avc,ind)
        Q_dist = np.delete(Q_dist,ind)

    return pwv_fus,pwv_avc,Q_dist

def outlier_with_sigma(data,thod=3):
    # 计算数据的均值和标准差
    mean = np.mean(data)
    std_dev = np.std(data)
    # 计算数据的3倍标准差
    threshold_upper = mean + thod * std_dev
    threshold_lower = mean - thod * std_dev

    # 剔除3倍中误差数据
    ind = np.where((data <= threshold_lower) | (data >= threshold_upper) )[0]
    return ind

def Robust_Z_score_Method_3sigma(data):
    # 计算R.Z.score
    med = np.median(data)
    mad = scipy.stats.median_abs_deviation(data)
    # r_z_score = data.map(lambda x: (0.6745*(x-med)) / (np.median(mad)))
    r_z_score = np.apply_along_axis(lambda x: (0.6745*(x-med)) / (np.median(mad)), 0, data)
    # 判定异常值
    # mark = np.abs(r_z_score) > 3 
    ind = np.where(np.abs(r_z_score) > 3 )[0]
    return ind


#基于极差法的异常检测方法
def out_range_method(data):
    data = np.array(data)
    if len(data)<3 :
        return []
    Q1, Q3 = np.percentile(data, [25,75])
    IQR = Q3 - Q1
    minimum = Q1 - 1.5 * IQR
    maximum = Q3 + 1.5 * IQR
    ind = np.where((data < minimum) | (data > maximum))[0]
    return ind

def accuracyEvaluate(y_true,y_pred,removeOutlier=True):
    R,BIAS,MAE,MRE,RMSE,KGE = 0.0,0.0,0.0,0.0,0.0,0.0
    y_true_collector = np.copy(y_true)
    y_pred_collector = np.copy(y_pred)

    ### 1. 去除nan值
    idx = np.where(np.isnan(y_true_collector))
    y_true_collector = np.delete(y_true_collector,idx,axis=0)
    y_pred_collector = np.delete(y_pred_collector,idx,axis=0)
    idx = np.where(np.isnan(y_pred_collector))
    y_true_collector = np.delete(y_true_collector,idx,axis=0)
    y_pred_collector = np.delete(y_pred_collector,idx,axis=0)
    if len(y_pred_collector) <= 1:
        return R,BIAS,MAE,MRE,RMSE,KGE
    
    ### 2. 剔除粗差
    dy = y_pred_collector - y_true_collector
    if removeOutlier:
        idx = outlier_with_sigma(dy)
        # idx = out_range_method(dy)
        dy_smooth = np.delete(dy,idx,axis=0)
        y_true_collector = np.delete(y_true_collector,idx,axis=0)
        y_pred_collector = np.delete(y_pred_collector,idx,axis=0)
    else:
        dy_smooth = dy
    
    if len(dy_smooth) <= 1:
        return R,BIAS,MAE,MRE,RMSE,KGE
    ### 3.1 求解皮尔逊相关系数（Pearson Correlation Coefficient）
    R = pearsonr(y_true_collector, y_pred_collector)[0]

    ### 3.2 求解总体偏差
    BIAS = np.mean(dy_smooth)

    ### 3.3 求解平均绝对误差
    MAE = np.sum(np.abs(dy_smooth))/len(dy_smooth)  

    ### 3.4 求解平均相对误差
    MRE = np.sum(np.abs(dy_smooth))/np.sum(np.abs(y_true_collector)) 

    ### 3.5 求解均方根误差
    RMSE = np.sqrt(mean_squared_error(y_true_collector, y_pred_collector))

    ### 3.6 KGE
    beta = np.mean(y_pred_collector)/np.mean(y_true_collector)
    gamma = np.std(y_pred_collector)/np.std(y_true_collector)
    KGE = 1 - np.sqrt((R-1)*(R-1) + (beta-1)*(beta-1) + (gamma-1)*(gamma-1))

    return R,BIAS,MAE,MRE,RMSE,KGE

def accuracyEvaluate_bySite(blyy):
    labels = blyy[:, 0]  
    unique_labels = np.unique(labels) 
    groups = {}
    for label in unique_labels:
        group = blyy[labels == label]
        groups[label] = group
    metric = []
    for key, value in groups.items():
        lat = key; lon = value[0,1]
        y_true = value[:,2]
        y_pred = value[:,3]
        R,BIAS,MAE,MRE,RMSE,KGE = accuracyEvaluate(y_true,y_pred,True)
        metric.append([lat,lon,R,BIAS,MAE,MRE,RMSE,KGE])
    metric = np.array(metric)
    return metric

def accuracyEvaluate_byTime(data):
    # 向下取整第三列（索引0，年积日）
    doy_floored = np.floor(data[:, 0]).astype(int)

    # 替换原数据中的年积日为取整后的值
    data[:, 0] = doy_floored

    # 按照年积日分组
    daily_data_dict = defaultdict(list)
    for row, day in zip(data, doy_floored):
        daily_data_dict[day].append(row)

    # 转换为 numpy 数组形式（可选）
    for day in daily_data_dict:
        daily_data_dict[day] = np.array(daily_data_dict[day])
    
    metric = []
    for day, rows in daily_data_dict.items():
        doy = day
        y_true = rows[:,1]
        y_pred = rows[:,2]
        meanTrue = np.nanmean(y_true)
        meanPred = np.nanmean(y_pred)
        R,BIAS,MAE,MRE,RMSE,KGE = accuracyEvaluate(y_true,y_pred,True)
        metric.append([doy,meanTrue,meanPred,R,BIAS,MAE,MRE,RMSE,KGE])
    metric = np.array(metric)
    return metric

def accuracyEvaluate_byHeightTop(data):
    # 获取第7列（高度数据）
    heights = data[:, 0]

    # 定义分段区间（例如从0到5000米，每500米一段）
    bins = np.arange(0, heights.max() + 500, 500)

    # 使用 np.digitize 获取每个高度属于哪个区间
    bin_indices = np.digitize(heights, bins)  # 注意：返回的是每个值属于的 bin 的索引（从1开始）

    # 分组：按高度区间分组数据
    grouped_data = {}
    for i, bin_idx in enumerate(bin_indices):
        bin_label = bins[bin_idx - 1]
        grouped_data.setdefault(bin_label, []).append(data[i])

    # 将列表转换为数组（可选）
    for k in grouped_data:
        grouped_data[k] = np.array(grouped_data[k])
    
    metric = []
    for key, rows in grouped_data.items():
        y_true = rows[:,1]
        y_pred = rows[:,2]
        R,BIAS,MAE,MRE,RMSE,KGE = accuracyEvaluate(y_true,y_pred,True)
        metric.append([key,R,BIAS,MAE,MRE,RMSE,KGE])
    metric = np.array(metric)
    return metric

def accuracyEvaluate_byTTop(data):
    # 获取第7列（高度数据）
    heights = data[:, 0]

    # 定义分段区间（例如从0到5000米，每500米一段）
    bins = np.arange(190, heights.max() + 10, 10)

    # 使用 np.digitize 获取每个高度属于哪个区间
    bin_indices = np.digitize(heights, bins)  # 注意：返回的是每个值属于的 bin 的索引（从1开始）

    # 分组：按高度区间分组数据
    grouped_data = {}
    for i, bin_idx in enumerate(bin_indices):
        bin_label = bins[bin_idx - 1]
        grouped_data.setdefault(bin_label, []).append(data[i])

    # 将列表转换为数组（可选）
    for k in grouped_data:
        grouped_data[k] = np.array(grouped_data[k])
    
    metric = []
    for key, rows in grouped_data.items():
        y_true = rows[:,1]
        y_pred = rows[:,2]
        R,BIAS,MAE,MRE,RMSE,KGE = accuracyEvaluate(y_true,y_pred,True)
        metric.append([key,R,BIAS,MAE,MRE,RMSE,KGE])
    metric = np.array(metric)
    return metric

def accuracyEvaluate_byPressTop(data):
    # 获取第7列（高度数据）
    heights = data[:, 0]

    # 定义分段区间（例如从0到5000米，每500米一段）
    bins = np.arange(100, heights.max() + 50, 50)

    # 使用 np.digitize 获取每个高度属于哪个区间
    bin_indices = np.digitize(heights, bins)  # 注意：返回的是每个值属于的 bin 的索引（从1开始）

    # 分组：按高度区间分组数据
    grouped_data = {}
    for i, bin_idx in enumerate(bin_indices):
        bin_label = bins[bin_idx - 1]
        grouped_data.setdefault(bin_label, []).append(data[i])

    # 将列表转换为数组（可选）
    for k in grouped_data:
        grouped_data[k] = np.array(grouped_data[k])
    
    metric = []
    for key, rows in grouped_data.items():
        y_true = rows[:,1]
        y_pred = rows[:,2]
        R,BIAS,MAE,MRE,RMSE,KGE = accuracyEvaluate(y_true,y_pred,True)
        metric.append([key,R,BIAS,MAE,MRE,RMSE,KGE])
    metric = np.array(metric)
    return metric


def accuracyEvaluate_byCloud(data,removeOutlier=True):
    # 将第1列（索引为0）提取出来作为分类依据
    labels = data[:, 0]

    # 获取唯一的类别值（例如 1、2、3、4）
    unique_labels = np.unique(labels)

    # # 按标签分割数据
    # grouped_data = {label: data[labels == label] for label in unique_labels}

    # 分组：1为一类，2、3、4合并为一类
    grouped_data = {
        'clear': data[labels == 1],
        'cloud': data[np.isin(labels, [2, 3, 4])]
    }

    # 将列表转换为数组
    for k in grouped_data:
        grouped_data[k] = np.array(grouped_data[k])
    
    metric = []
    for key, rows in grouped_data.items():
        y_true = rows[:,1]
        y_pred = rows[:,2]
        R,BIAS,MAE,MRE,RMSE,KGE = accuracyEvaluate(y_true,y_pred,removeOutlier)
        metric.append([key,R,BIAS,MAE,MRE,RMSE,KGE])
    metric = np.array(metric)
    return metric

def readH5py(h5_file,readTime=True):
    with h5py.File(h5_file, "r") as hf:  # read data from h5 file
        data = hf["X"][:]
        lat = hf["latitude"][:]
        lon = hf["longitude"][:]
        if readTime:
            time = hf["time"][:]
        else:
            time = []
    return data,lat,lon,time

def writeH5py(h5_file,data,lat,lon,time=[]):
    with h5py.File(h5_file, "w") as hf:
        hf.create_dataset("X", data=np.array(data,dtype=np.float32))
        hf.create_dataset("latitude", data=np.array(lat,dtype=np.float32))
        hf.create_dataset("longitude", data=np.array(lon,dtype=np.float32))
        hf.create_dataset("time", data=time)

# def merge_csv(csv_files,out_csv=None):
#     df_all = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
#     merge_data = df_all.to_numpy()

#     if out_csv != None:
#         df_all.to_csv(out_csv, index=False, encoding='utf-8-sig')
#     return merge_data


class Interpolator:
    """
    Efficient DEM interpolator that loads DEM data into memory and supports
    nearest, linear, and cubic interpolation for given lon/lat grids.
    """

    def __init__(self, tif_path):
        """
        Load DEM into memory once.
        """
        with rasterio.open(tif_path) as dem:
            self.dem_data = dem.read(1).astype(float)
            self.transform = dem.transform
            self.crs = dem.crs
            self.nodata = dem.nodata if dem.nodata is not None else np.nan

        # 反向仿射变换（地理 -> 像素坐标）
        self.inv_affine = ~self.transform

        print(f"✅ DEM loaded into memory: shape={self.dem_data.shape}, CRS={self.crs}")

    def interpolate(self, lon, lat, method="linear", fill_value=np.nan):
        """
        Interpolate DEM values at given lon/lat using specified interpolation method.

        Parameters
        ----------
        lon : np.ndarray
            1D or 2D array of longitudes.
        lat : np.ndarray
            1D or 2D array of latitudes.
        method : str, optional
            Interpolation method: 'nearest', 'linear', or 'cubic'. Default is 'linear'.
        fill_value : float, optional
            Value to assign for points outside DEM bounds.

        Returns
        -------
        dem_interp : np.ndarray
            Interpolated DEM values with same shape as input lon/lat.
        """
        if lon.ndim == 1 and lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)

        # 将经纬度转换为行列号（像素坐标）
        col, row = self.inv_affine * (lon, lat)

        # 边界检查
        nrows, ncols = self.dem_data.shape
        inside = (row >= 0) & (row < nrows - 1) & (col >= 0) & (col < ncols - 1)

        # 创建结果数组
        result = np.full(lon.shape, fill_value, dtype=float)

        # 插值方式选择
        if method == "nearest":
            row_i = np.round(row).astype(int)
            col_i = np.round(col).astype(int)
            valid = (row_i >= 0) & (row_i < nrows) & (col_i >= 0) & (col_i < ncols)
            result[valid] = self.dem_data[row_i[valid], col_i[valid]]

        elif method == "linear":
            # 双线性插值
            row0 = np.floor(row).astype(int)
            col0 = np.floor(col).astype(int)
            row1 = row0 + 1
            col1 = col0 + 1

            w_row = row - row0
            w_col = col - col0

            valid = inside.copy()
            f00 = self.dem_data[row0, col0]
            f10 = self.dem_data[row0, col1]
            f01 = self.dem_data[row1, col0]
            f11 = self.dem_data[row1, col1]

            interp_val = (
                f00 * (1 - w_row) * (1 - w_col)
                + f10 * (1 - w_row) * w_col
                + f01 * w_row * (1 - w_col)
                + f11 * w_row * w_col
            )
            result[valid] = interp_val[valid]

        elif method == "cubic":
            # 使用 scipy.ndimage.map_coordinates 实现双三次插值
            coords = np.vstack((row.ravel(), col.ravel()))
            interp_val = map_coordinates(
                self.dem_data,
                coords,
                order=3,  # 三次插值
                mode='nearest',
                cval=fill_value
            )
            result = interp_val.reshape(lon.shape)

        else:
            raise ValueError("method must be one of: 'nearest', 'linear', 'cubic'")

        return result
    