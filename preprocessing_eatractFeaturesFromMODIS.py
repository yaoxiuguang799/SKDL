import os
from scipy.io import savemat,loadmat
from pyhdf.SD import SD, SDC
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from osgeo import gdal
from tqdm import tqdm
from multiprocessing import Pool,Manager,TimeoutError
from joblib import Parallel, delayed
from global_tropopause_model.bth_model import bth_model

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

# 读取DEM数据
def read_tif(dem_file):
    tif_dataset = gdal.Open(dem_file)
    if tif_dataset is None:
        print("无法打开tif文件")
        return None
    geotransform = tif_dataset.GetGeoTransform()
    band = tif_dataset.GetRasterBand(1)  
    dataArray = band.ReadAsArray()
    return geotransform,dataArray
 
# 插值DEM数据
def interpolate_tif(geotransform, dataArray,x, y):
    x_offset = geotransform[0]
    y_offset = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
 
    # 将地面坐标转换为像素坐标
    pixel = int((x - x_offset) / pixel_width)
    line = int((y - y_offset) / pixel_height)
 
    # 进行插值
    interpolated_value = -9999
    if pixel >= 0 and line >= 0:
        try:
            # 使用默认的插值方法，这里可以选择其他插值方法，如最邻近、双线性等
            # interpolated_value = tif_band.ReadAsArray(pixel, line, 1, 1)[0][0]
            interpolated_value = dataArray[line, pixel]
        except Exception as e:
            print(f"插值失败: {e}")
    interpolated_value = float(interpolated_value)
    return interpolated_value

def compute_cloud_mask(Cloud_Mask_QA):        
        # 第一字节的数据
        byte1_data = Cloud_Mask_QA
        # 比特0表示为否进行了检测, determined or not
        bits0and1 = byte1_data & 0b00000001
        determined = bits0and1.astype(int)
        # 第1、2比特表示置信度,00为云, 01为不确定, 10为可能晴空, 11为确信晴空
        #Cloudy, Uncertainty, Probably Clear, Confident Clear
        bits1and2 = byte1_data & 0b00000110
        
        cloud_mask_list = []
        cloud_mask_list.append(determined)
        
        cloudy = (bits1and2 < 2).astype(int)
        cloud_mask_list.append(cloudy)
        
        uncertain = (bits1and2 == 2).astype(int)
        cloud_mask_list.append(uncertain)
        
        probably_clear = (bits1and2 == 4).astype(int)
        cloud_mask_list.append(probably_clear)
        
        confident_clear = (bits1and2 == 6).astype(int)
        cloud_mask_list.append(confident_clear)
        
        cloud_mask = np.stack(cloud_mask_list, axis=2)
        
        return cloud_mask

def compute_qa(QA_index):        
        # 第一字节的数据
        byte1_data = QA_index
        # 比特0表示为否进行了检测, determined or not
        # 第1、2比特表示置信度,00为云, 01为不确定, 10为可能晴空, 11为确信晴空
        #Cloudy, Uncertainty, Probably Clear, Confident Clear
        bits1and2 = byte1_data & 0b00000110
        
        qa_list = []   

        #  00为Marginal Confidence     
        Marginal_Confidence1 = (bits1and2 < 2).astype(int)
        qa_list.append(Marginal_Confidence1)
        
        # %01为Marginal Confidence
        Marginal_Confidence2 = (bits1and2 == 2).astype(int)
        qa_list.append(Marginal_Confidence2)
        
        # 10为Good Confidence
        Good_Confidence = (bits1and2 == 4).astype(int)
        qa_list.append(Good_Confidence)
        
        # 11为Very Good Confidence
        Very_Good_Confidence = (bits1and2 == 6).astype(int)
        qa_list.append(Very_Good_Confidence)
        
        qa_list = np.stack(qa_list, axis=2)
        
        return qa_list

def space_match(lat_mod3,lon_mod3,lat_points,lon_points):
        max_lat_mod3 = np.max(lat_mod3);min_lat_mod3 = np.min(lat_mod3)
        max_lon_mod3 = np.max(lon_mod3);min_lon_mod3 = np.min(lon_mod3)
        min_dis = []
        indexs_rs = []
        indexs_bl = []

        for idx in range(len(lat_points)):
            lat = lat_points[idx]
            lon = lon_points[idx]
            if lat>max_lat_mod3 or lat<min_lat_mod3 or lon>max_lon_mod3 or lon<min_lon_mod3:
                continue

            dist_to_point = np.sqrt((lat_mod3 - lat)**2 + (lon_mod3 - lon)**2)
            nearest_point = np.unravel_index(dist_to_point.argmin(),dist_to_point.shape)
            nearest_dist = dist_to_point[nearest_point]
            if nearest_dist > 0.05:
                continue
            min_dis.append(nearest_dist)
            indexs_rs.append(nearest_point)
            indexs_bl.append(idx)
        return indexs_rs,indexs_bl,min_dis

# def processSingleScene(MOD03,path_MOD021KM,path_MOD05_L2,path_MOD06_L2,gnss_lat,gnss_lon,gnss_data,igra_lat,igra_lon,igra_data,pvu=3.5):
def processSingleScene(MOD03):
    # MOD03 = 'MODIS\\MOD03\\MOD03.A2020009.1005.061.2020009170524.hdf'
    names_gnss,dataset_gnss,names_igra,dataset_igra = [],[],[],[]
    basename = os.path.basename(MOD03)
    type = basename[0:3]
    name_time = basename[6:23]
    print('%s:正在处理.........'%(name_time))
    if type == 'MOD':
        MOD05 = glob(path_MOD05_L2 + '\\MOD05_L2.%s*.hdf'%(name_time))
        MOD06 = glob(path_MOD06_L2 + '\\MOD06_L2.%s*.hdf'%(name_time))
        MOD021KM = glob(path_MOD021KM + '\\MOD021KM.%s*.hdf'%(name_time))
    elif type == 'MYD':
        MOD05 = glob(path_MOD05_L2 + '\\MYD05_L2.%s*.hdf'%(name_time))
        MOD06 = glob(path_MOD06_L2 + '\\MYD06_L2.%s*.hdf'%(name_time))
        MOD021KM = glob(path_MOD021KM + '\\MYD021KM.%s*.hdf'%(name_time))
    if len(MOD05) == 0 or  len(MOD06) == 0 or len(MOD021KM) == 0:
        return names_gnss,dataset_gnss,names_igra,dataset_igra
    MOD03_size = os.path.getsize(MOD03)/1024.0
    MOD05_size = os.path.getsize(MOD05[0])/1024.0
    MOD06_size = os.path.getsize(MOD06[0])/1024.0
    MOD021KM_size = os.path.getsize(MOD021KM[0])/1024.0
    if MOD03_size < 100 or MOD05_size < 100 or MOD06_size < 100 or MOD021KM_size < 100:
        return names_gnss,dataset_gnss,names_igra,dataset_igra
    ### 解析遥感观测时间
    doy = float(name_time[5:8])
    hour = float(name_time[9:11])
    minute = float(name_time[11:13])
    doyf = doy+hour/24.0+minute/1440.0

    ### 读取MOD03文件，获取lat,lon,SensorAzimuth,SensorZenith,SolarAzimuth,SolarZenith,Range
    try:
        hdf_mod3 = SD( MOD03, SDC.READ )
        datasets_dic = hdf_mod3.datasets()
        lat_mod3 = hdf_mod3.select('Latitude').get()  # degrees
        lon_mod3 = hdf_mod3.select('Longitude').get() # degrees
        Height = hdf_mod3.select('Height').get() # Height above geoid ，meters
        Height = Height*1.0
        Height[Height==-32767] = np.nan
        SensorAzimuth = hdf_mod3.select('SensorAzimuth').get()
        SensorAzimuth = SensorAzimuth*0.01           # degrees
        SensorAzimuth[SensorAzimuth==-327.67] = np.nan          
        SensorZenith = hdf_mod3.select('SensorZenith').get()
        SensorZenith = SensorZenith*0.01              # degrees
        SensorZenith[SensorZenith==-327.67] = np.nan
        SolarAzimuth = hdf_mod3.select('SolarAzimuth').get()
        SolarAzimuth = SolarAzimuth*0.01              # degrees
        SolarAzimuth[SolarAzimuth==-327.67] = np.nan
        SolarZenith = hdf_mod3.select('SolarZenith').get()
        SolarZenith = SolarZenith*0.01                # degrees
        SolarZenith[SolarZenith==-327.67] = np.nan
        Range = hdf_mod3.select('Range').get()
        Range = Range*25.0                            # meters
        Range[Range==0] = np.nan
        hdf_mod3.end()
    except Exception as e:
        # 出现错误时的处理逻辑
        print('%s:文件读取错误'%(MOD03))
        print(f"Error: {e}")
        return names_gnss,dataset_gnss,names_igra,dataset_igra
    
    size_mod3_x,size_mod3_y = lat_mod3.shape

    indexs_rs_gnss,indexs_bl_gnss,min_dis_gnss = space_match(lat_mod3,lon_mod3,gnss_lat,gnss_lon)
    indexs_rs_igra,indexs_bl_igra,min_dis_igra = space_match(lat_mod3,lon_mod3,igra_lat,igra_lon)
    min_dis_igra = []
    print('地面站点匹配情况：GNSS:%s；IGRAl:%s'%(len(min_dis_gnss),len(min_dis_igra)))
    # 若该景影像不包含任何地面站点，不做任何处理。
    if len(min_dis_gnss) == 0 and len(min_dis_igra) == 0:
        return names_gnss,dataset_gnss,names_igra,dataset_igra


    ## 读取MOD021KM文件，获取
    
    try:
        hdf_MOD021KM = SD( MOD021KM[0], SDC.READ )
        datasets_dic = hdf_MOD021KM.datasets()
        RefSB = hdf_MOD021KM.select('EV_1KM_RefSB')
        data = RefSB.get()
        attrs = RefSB.attributes()
        reflectance_scales = attrs['reflectance_scales']
        reflectance_offsets = attrs['reflectance_offsets']
        RefSB_250 = hdf_MOD021KM.select('EV_250_Aggr1km_RefSB')
        data_250 = RefSB_250.get()
        attrs_250 = RefSB_250.attributes()
        reflectance_scales_250 = attrs_250['reflectance_scales']
        reflectance_offsets_250 = attrs_250['reflectance_offsets']
        RefSB_500 = hdf_MOD021KM.select('EV_500_Aggr1km_RefSB')
        data_500 = RefSB_500.get()
        attrs_500 = RefSB_500.attributes()
        reflectance_scales_500 = attrs_500['reflectance_scales']
        reflectance_offsets_500 = attrs_500['reflectance_offsets']
        tmp,size_mod2_x,size_mod2_y = data.shape
        hdf_MOD021KM.end()
    except Exception as e:
        # 出现错误时的处理逻辑
        print('%s:文件读取错误'%(MOD021KM[0]))
        print(f"Error: {e}")
        return names_gnss,dataset_gnss,names_igra,dataset_igra
    

    ### 若MOD021KM的维度与MOD03不一致，无法保证数据匹配，不做处理
    if size_mod2_x != size_mod3_x or size_mod2_y != size_mod3_y:
        return names_gnss,dataset_gnss,names_igra,dataset_igra

    Band02_scale = reflectance_scales_250[1];Band02_offsets = reflectance_offsets_250[1];Band02_RefSB = data_250[1,:,:]*1.0
    Band05_scale = reflectance_scales_500[2];Band05_offsets = reflectance_offsets_500[2];Band05_RefSB = data_500[2,:,:]*1.0
    Band17_scale = reflectance_scales[11];Band18_scale = reflectance_scales[12];Band19_scale = reflectance_scales[13]
    Band17_offsets = reflectance_offsets[11];Band18_offsets = reflectance_offsets[12];Band19_offsets = reflectance_offsets[13]
    Band17_RefSB = data[11,:,:]*1.0;Band18_RefSB = data[12,:,:]*1.0;Band19_RefSB = data[13,:,:]*1.0

    Band02_RefSB[Band02_RefSB==65535] = np.nan;Band05_RefSB[Band05_RefSB==65535] = np.nan
    Band17_RefSB[Band17_RefSB==65535] = np.nan;Band18_RefSB[Band18_RefSB==65535] = np.nan;Band19_RefSB[Band19_RefSB==65535] = np.nan
    
    Band02_RefSB = (Band02_RefSB-Band02_offsets)*Band02_scale
    Band05_RefSB = (Band05_RefSB-Band05_offsets)*Band05_scale
    Band17_RefSB = (Band17_RefSB-Band17_offsets)*Band17_scale
    Band18_RefSB = (Band18_RefSB-Band18_offsets)*Band18_scale
    Band19_RefSB = (Band19_RefSB-Band19_offsets)*Band19_scale
    

    ## 读取MOD06_L2文件，获取Tropopause_Height,cloud_top_height_1km,cloud_top_pressure_1km,cloud_top_temperature_1km
    
    try:
        hdf_mod6 = SD( MOD06[0], SDC.READ )
        datasets_dic = hdf_mod6.datasets()
        Tropopause_Height = hdf_mod6.select('Tropopause_Height').get()     
        Tropopause_Height = Tropopause_Height*0.1                                               # hPa                         
        Tropopause_Height[Tropopause_Height==-3276.8] = np.nan
        cloud_top_height_1km = hdf_mod6.select('cloud_top_height_1km').get()        
        cloud_top_height_1km = cloud_top_height_1km*1.0                
        cloud_top_height_1km[cloud_top_height_1km==-999] = np.nan                               # meters
        cloud_top_pressure_1km = hdf_mod6.select('cloud_top_pressure_1km').get()     
        cloud_top_pressure_1km = cloud_top_pressure_1km*1.0                                     # hPa
        cloud_top_pressure_1km[cloud_top_pressure_1km==-999] = np.nan  
        cloud_top_pressure_1km = cloud_top_pressure_1km*0.1
        Cloud_Optical_Thickness = hdf_mod6.select('Cloud_Optical_Thickness').get() 
        Cloud_Optical_Thickness = Cloud_Optical_Thickness*1.0
        Cloud_Optical_Thickness[Cloud_Optical_Thickness==-9999] = np.nan                       #none
        Cloud_Optical_Thickness = Cloud_Optical_Thickness*0.01
        Cloud_Optical_Thickness_PCL = hdf_mod6.select('Cloud_Optical_Thickness_PCL').get() 
        Cloud_Optical_Thickness_PCL = Cloud_Optical_Thickness_PCL*1.0
        Cloud_Optical_Thickness_PCL[Cloud_Optical_Thickness_PCL==-9999] = np.nan 
        Cloud_Optical_Thickness_PCL = Cloud_Optical_Thickness_PCL*0.01
        Cloud_Water_Path = hdf_mod6.select('Cloud_Water_Path').get()
        Cloud_Water_Path = Cloud_Water_Path*1.0
        Cloud_Water_Path[Cloud_Water_Path==-9999] = np.nan                                      # g/m^2 
        Cloud_Water_Path_PCL = hdf_mod6.select('Cloud_Water_Path_PCL').get()
        Cloud_Water_Path_PCL = Cloud_Water_Path_PCL*1.0
        Cloud_Water_Path_PCL[Cloud_Water_Path_PCL==-9999] = np.nan
        surface_temperature_1km = hdf_mod6.select('surface_temperature_1km').get()
        surface_temperature_1km = surface_temperature_1km*1.0
        surface_temperature_1km[surface_temperature_1km==-999] = np.nan
        surface_temperature_1km = (surface_temperature_1km + 15000)*0.01    
        cloud_top_temperature_1km = hdf_mod6.select('cloud_top_temperature_1km').get()
        cloud_top_temperature_1km = cloud_top_temperature_1km*1.0
        cloud_top_temperature_1km[cloud_top_temperature_1km==-999] = np.nan
        cloud_top_temperature_1km = (cloud_top_temperature_1km + 15000)*0.01                     # K
        size_mod6_x,size_mod6_y = cloud_top_height_1km.shape
        hdf_mod6.end()
    except Exception as e:
        # 出现错误时的处理逻辑
        print('%s:文件读取错误'%(MOD06[0]))
        print(f"Error: {e}")
        return names_gnss,dataset_gnss,names_igra,dataset_igra
    
    ### 若MOD06_L2的维度与MOD03不一致，无法保证数据匹配，不做处理
    if size_mod6_x != size_mod3_x or size_mod6_y != size_mod3_y:
        return names_gnss,dataset_gnss,names_igra,dataset_igra

    ## 读取MOD05_L2文件，获取Water_Vapor_Near_Infrared,Cloud_Mask_QA,Quality_Assurance_Near_Infrared
    
    try:
        hdf_mod5 = SD( MOD05[0], SDC.READ )
        datasets_dic = hdf_mod5.datasets()
        PwvNir = hdf_mod5.select('Water_Vapor_Near_Infrared').get()
        PwvNir = PwvNir*1.0
        PwvNir[PwvNir==-9999] = np.nan
        PwvNir = PwvNir*0.01
        Cloud_Mask_QA = hdf_mod5.select('Cloud_Mask_QA').get()
        Quality_Assurance_Near_Infrared = hdf_mod5.select('Quality_Assurance_Near_Infrared').get()
        Quality_Assurance_Near_Infrared = Quality_Assurance_Near_Infrared.reshape((Quality_Assurance_Near_Infrared.shape[0],Quality_Assurance_Near_Infrared.shape[1]))
        QANI = compute_qa(Quality_Assurance_Near_Infrared)
        # sum1 = np.sum(QANI[:,:,0])
        # sum2 = np.sum(QANI[:,:,1])
        # sum3 = np.sum(QANI[:,:,2])
        # sum4 = np.sum(QANI[:,:,3])
        cloud_mask = compute_cloud_mask(Cloud_Mask_QA)
        size_mod5_x,size_mod5_y = PwvNir.shape
        hdf_mod5.end()
    except Exception as e:
        # 出现错误时的处理逻辑
        print('%s:文件读取错误'%(MOD05[0]))
        print(f"Error: {e}")
        return names_gnss,dataset_gnss,names_igra,dataset_igra
    

    ### 提取GNSS站点处特征变量
    for idx in range(len(min_dis_gnss)):
        idx_rs = indexs_rs_gnss[idx]
        site_idx = indexs_bl_gnss[idx]
        min_dis = min_dis_gnss[idx]
        site_name = gnss_data['sta_name'][site_idx]
        site_times = gnss_data[site_name][:,0]
        site_pwvs = gnss_data[site_name][:,1]

        neareat_idx,neareatTime = find_nearest(site_times,doyf)
        ### 如果最接近的时间与遥感观测时间差大于30分钟，则不处理
        min_time = np.abs(doyf-neareatTime)*1440
        if min_time > 30.0:
            continue
        

        ### GNSS site 
        lat = float(gnss_lat[site_idx])
        lon = float(gnss_lon[site_idx])
        site_pwv = float(site_pwvs[neareat_idx])
        height_geodetic = float(gnss_data['height'][0,site_idx])

        ### SRTM dem 正常高
        if lat>=84 or lat<-56:
            height_geoid = -9999
        else:
            height_geoid = interpolate_tif(geotransform_dem, dataArray_dem,lon, lat)


        ### time 
        time = doyf

        ### MOD03 geolocation
        SensorA = SensorAzimuth[idx_rs]
        SensorZ = SensorZenith[idx_rs]
        SolarA  = SolarAzimuth[idx_rs]
        SolarZ  = SolarZenith[idx_rs]
        # height_geodetic = Height[idx_rs]
        RangeToSensor = Range[idx_rs]

        ### MOD021KM
        Band02_rf = Band02_RefSB[idx_rs]
        Band05_rf = Band05_RefSB[idx_rs]
        Band17_rf = Band17_RefSB[idx_rs]
        Band18_rf = Band18_RefSB[idx_rs]
        Band19_rf = Band19_RefSB[idx_rs]

        ### MOD06_L2
        cloud_top_height_idx = cloud_top_height_1km[idx_rs]
        cloud_top_pressure_idx = cloud_top_pressure_1km[idx_rs]
        cloud_top_temperature_idx = cloud_top_temperature_1km[idx_rs]
        Cloud_Optical_Thickness_idx = Cloud_Optical_Thickness[idx_rs]
        Cloud_Optical_Thickness_PCL_idx = Cloud_Optical_Thickness_PCL[idx_rs]
        Cloud_Water_Path_idx = Cloud_Water_Path[idx_rs]
        Cloud_Water_Path_PCL_idx = Cloud_Water_Path_PCL[idx_rs]
        surface_temperature_1km_idx = surface_temperature_1km[idx_rs]
    
        ### MOD05_L2
        
        if idx_rs[0] < size_mod5_x and idx_rs[1] < size_mod5_y:
            mod_nir_pwv = PwvNir[idx_rs]
            is_check = cloud_mask[idx_rs[0],idx_rs[1],0]; c_clear = cloud_mask[idx_rs[0],idx_rs[1],4];p_clear = cloud_mask[idx_rs[0],idx_rs[1],3]
            p_cloud = cloud_mask[idx_rs[0],idx_rs[1],2]; c_cloud = cloud_mask[idx_rs[0],idx_rs[1],1]
            vg_Confidence = QANI[idx_rs[0],idx_rs[1],3];g_Confidence = QANI[idx_rs[0],idx_rs[1],2];m_ConfidenceOver60 = QANI[idx_rs[0],idx_rs[1],1];m_ConfidenceDown60 = QANI[idx_rs[0],idx_rs[1],0]
            if not is_check:
                mod_nir_pwv = np.nan
                cmask = 0
            else:
                if c_clear:
                    cmask = 1.0
                elif p_clear:
                    cmask = 2.0
                elif p_cloud:
                    cmask = 3.0
                elif c_cloud:
                    cmask = 4.0

            if vg_Confidence:
                qa = 1.0
            elif g_Confidence:
                qa = 2.0
            elif m_ConfidenceOver60:
                qa = 3.0
            elif m_ConfidenceDown60:
                qa = 4.0
        else:
            mod_nir_pwv = np.nan
            cmask = 0
            qa = 0
        
        ### MCD12Q1
        lc = interpolate_tif(geotransform_lc, dataArray_lc,lon, lat)

        ### tropopause height
        z = bth_model(lat, int(doy), pvu)
        tropopause_height = float(z[0])*1000.0

        ### create out vector [lat,lon,height_geodetic,height_geoid,time,]
        dataset = [lat,lon,height_geodetic,height_geoid,time,
                Band02_rf,Band05_rf,Band17_rf,Band18_rf,Band19_rf,
                SensorA,SensorZ,SolarA,SolarZ,RangeToSensor,
                cloud_top_pressure_idx,cloud_top_temperature_idx,Cloud_Optical_Thickness_idx,
                Cloud_Optical_Thickness_PCL_idx,Cloud_Water_Path_idx,Cloud_Water_Path_PCL_idx,surface_temperature_1km_idx,cloud_top_height_idx,tropopause_height,
                lc,cmask,qa,mod_nir_pwv,site_pwv,min_dis,min_time]
        names_gnss.append(site_name)
        dataset_gnss.append(dataset)
        
         ### 提取探空站点处特征变量

    for idx in range(len(min_dis_igra)):
        idx_rs = indexs_rs_igra[idx]
        site_idx = indexs_bl_igra[idx]
        min_dis = min_dis_igra[idx]
        site_name = igra_data['stationID'][site_idx]
        site_times = igra_data[site_name][:,0]
        site_pwvs = igra_data[site_name][:,1]

        neareat_idx,neareatTime = find_nearest(site_times,doyf)
        ### 如果最接近的时间与遥感观测时间差大于30分钟，则不处理
        min_time = np.abs(doyf-neareatTime)*1440
        if min_time > 60.0:
            continue
        

        ### IGRA site 
        lat = float(igra_lat[site_idx])
        lon = float(igra_lon[site_idx])
        site_pwv = float(site_pwvs[neareat_idx])

        ### SRTM dem 正常高
        if lat>=84 or lat<-56:
            height_geoid = -9999
        else:
            height_geoid = interpolate_tif(geotransform_dem, dataArray_dem,lon, lat)

        ### time 
        time = doyf

        ### MOD03 geolocation
        SensorA = SensorAzimuth[idx_rs]
        SensorZ = SensorZenith[idx_rs]
        SolarA  = SolarAzimuth[idx_rs]
        SolarZ  = SolarZenith[idx_rs]
        height_geodetic = Height[idx_rs]
        RangeToSensor = Range[idx_rs]

        ### MOD021KM
        Band02_rf = Band02_RefSB[idx_rs]
        Band05_rf = Band05_RefSB[idx_rs]
        Band17_rf = Band17_RefSB[idx_rs]
        Band18_rf = Band18_RefSB[idx_rs]
        Band19_rf = Band19_RefSB[idx_rs]

        ### MOD06_L2
        cloud_top_height_idx = cloud_top_height_1km[idx_rs]
        cloud_top_pressure_idx = cloud_top_pressure_1km[idx_rs]
        cloud_top_temperature_idx = cloud_top_temperature_1km[idx_rs]
        Cloud_Optical_Thickness_idx = Cloud_Optical_Thickness[idx_rs]
        Cloud_Optical_Thickness_PCL_idx = Cloud_Optical_Thickness_PCL[idx_rs]
        Cloud_Water_Path_idx = Cloud_Water_Path[idx_rs]
        Cloud_Water_Path_PCL_idx = Cloud_Water_Path_PCL[idx_rs]
        surface_temperature_1km_idx = surface_temperature_1km[idx_rs]

        ### MOD05_L2
        size_mod5_x,size_mod5_y
        if idx_rs[0] < size_mod5_x and idx_rs[1] < size_mod5_y:
            mod_nir_pwv = PwvNir[idx_rs]
            is_check = cloud_mask[idx_rs[0],idx_rs[1],0]; c_clear = cloud_mask[idx_rs[0],idx_rs[1],4];p_clear = cloud_mask[idx_rs[0],idx_rs[1],3]
            p_cloud = cloud_mask[idx_rs[0],idx_rs[1],2]; c_cloud = cloud_mask[idx_rs[0],idx_rs[1],1]
            vg_Confidence = QANI[idx_rs[0],idx_rs[1],3];g_Confidence = QANI[idx_rs[0],idx_rs[1],2];m_ConfidenceOver60 = QANI[idx_rs[0],idx_rs[1],1];m_ConfidenceDown60 = QANI[idx_rs[0],idx_rs[1],0]
            if not is_check:
                mod_nir_pwv = np.nan
                cmask = 0
            else:
                if c_clear:
                    cmask = 1.0
                elif p_clear:
                    cmask = 2.0
                elif p_cloud:
                    cmask = 3.0
                elif c_cloud:
                    cmask = 4.0

            if vg_Confidence:
                qa = 1.0
            elif g_Confidence:
                qa = 2.0
            elif m_ConfidenceOver60:
                qa = 3.0
            elif m_ConfidenceDown60:
                qa = 4.0
        else:
            mod_nir_pwv = np.nan
            cmask = 0
            qa = 0
         
        ### MCD12Q1
        lc = interpolate_tif(geotransform_lc, dataArray_lc,lon, lat)

        ### tropopause height
        z = bth_model(lat, int(doy), pvu)
        tropopause_height = float(z[0])*1000.0

        ### create out vector [lat,lon,height_geodetic,height_geoid,time,]
        dataset = [lat,lon,height_geodetic,height_geoid,time,
                Band02_rf,Band05_rf,Band17_rf,Band18_rf,Band19_rf,
                SensorA,SensorZ,SolarA,SolarZ,RangeToSensor,
                cloud_top_pressure_idx,cloud_top_temperature_idx,Cloud_Optical_Thickness_idx,
                Cloud_Optical_Thickness_PCL_idx,Cloud_Water_Path_idx,Cloud_Water_Path_PCL_idx,surface_temperature_1km_idx,cloud_top_height_idx,tropopause_height,
                lc,cmask,qa,mod_nir_pwv,site_pwv,min_dis,min_time]
        
        names_igra.append(site_name)
        dataset_igra.append(dataset)

    return names_gnss,dataset_gnss,names_igra,dataset_igra    

# 空间滤波处理 多进程函数
def MODIS_Extract_Handler(MOD03):
    names_gnss,dataset_gnss,names_igra,dataset_igra = processSingleScene(MOD03)
    return (names_gnss,dataset_gnss,names_igra,dataset_igra)

# 多进程：空间滤波处理
def ModisExtractMultThread(MOD03s,njob=8):
    n = len(MOD03s)
    result_list = Parallel(n_jobs=njob,prefer="threads")(delayed(MODIS_Extract_Handler)(MOD03s[i]) for i in tqdm(range(n)))
    datasets_gnss = {}
    datasets_igra = {}
    for result in result_list:
        names_gnss,dataset_gnss,names_igra,dataset_igra = result[0],result[1],result[2],result[3]
        for idx in range(len(names_gnss)):
            site_name = names_gnss[idx]
            dataset = dataset_gnss[idx]
            if not site_name in datasets_gnss:
                datasets_gnss[site_name] = []
                datasets_gnss[site_name].append(dataset)
            else:
                datasets_gnss[site_name].append(dataset)

    
        for idx in range(len(names_igra)):
            site_name = names_igra[idx]
            dataset = dataset_igra[idx]    
            if not site_name in datasets_igra:
                datasets_igra[site_name] = []
                datasets_igra[site_name].append(dataset)
            else:
                datasets_igra[site_name].append(dataset)

    return datasets_gnss,datasets_igra

def ModisExtractSingleThread(MOD03s):
    datasets_gnss = {}
    datasets_igra = {}
    for MOD03 in tqdm(MOD03s):
        names_gnss,dataset_gnss,names_igra,dataset_igra = processSingleScene(MOD03)
        
        for idx in range(len(names_gnss)):
            site_name = names_gnss[idx]
            dataset = dataset_gnss[idx]
            if not site_name in datasets_gnss:
                datasets_gnss[site_name] = []
                datasets_gnss[site_name].append(dataset)
            else:
                datasets_gnss[site_name].append(dataset)

    
        for idx in range(len(names_igra)):
            site_name = names_igra[idx]
            dataset = dataset_igra[idx]    
            if not site_name in datasets_igra:
                datasets_igra[site_name] = []
                datasets_igra[site_name].append(dataset)
            else:
                datasets_igra[site_name].append(dataset)
    return datasets_gnss,datasets_igra



####### MAIN 
pvu = 3.5; 

igra_file = '.\\IGRA_PWV\\IGRA_PWV_2020.mat'
gnss_file = '.\\gnss_pwv\\global\\gnss_pwv_west.mat'
dem_file = '.\\dem\\SRTM_Global_1km.tif'
landUse_file = '.\\MODIS\\MCD12Q1\\landuse_2020_1km.tiff'
path_MOD03 = '.\\MODIS\\MOD03'
path_MOD05_L2 = '.\\MODIS\\MOD05_L2'
path_MOD06_L2 = '.\\MODIS\\MOD06_L2'
path_MOD021KM = '.\\MODIS\\MOD021KM'
path_dataset_gnss = '.\\dataset\\gnss'
path_dataset_igra = '.\\dataset\\igra'

igra_data = loadmat(igra_file)
gnss_data = loadmat(gnss_file)
geotransform_lc,dataArray_lc = read_tif(landUse_file)
geotransform_dem,dataArray_dem = read_tif(dem_file)

igra_lat = igra_data['lat'].flatten()
igra_lon = igra_data['lon'].flatten()
gnss_coord = gnss_data['lat_lon']
gnss_lat = gnss_coord[:,0];gnss_lon = gnss_coord[:,1]

####开始处理
MOD03s = glob(path_MOD03 + '\\*.hdf')
MOD03s.sort()

### 提取截止年积日
time_strs = []
for MOD03 in MOD03s:
    basename = os.path.basename(MOD03)
    name_time = basename[6:23]
    ### 解析遥感观测时间
    doy = int(name_time[5:8])
    time_strs.append(doy)
time_strs = list(set(time_strs))
time_strs.sort()
doy_1 = time_strs[0]
doy_2 = time_strs[-1]
name_month = '_{:0>3d}_{:0>3d}'.format(doy_1,doy_2)

### 开始处理数据
MultThread = True
if not MultThread:
    datasets_gnss,datasets_igra = ModisExtractSingleThread(MOD03s)
else:
    datasets_gnss,datasets_igra = ModisExtractMultThread(MOD03s)


### 输出数据
for key,value in datasets_gnss.items():
    outFile = os.path.join(path_dataset_gnss,'%s%s_west.csv'%(key,name_month))
    dataset = np.asarray(value)
    dataset = dataset[dataset[:,4].argsort()] #按照第3列对行排序
    np.savetxt(outFile, dataset, delimiter=',', fmt='%.10f')

for key,value in datasets_igra.items():
    outFile = os.path.join(path_dataset_igra,'%s%s_west.csv'%(key,name_month))
    dataset = np.asarray(value)
    dataset = dataset[dataset[:,4].argsort()] #按照第3列对行排序
    np.savetxt(outFile, dataset, delimiter=',', fmt='%.10f')

print('all is over!!!')


